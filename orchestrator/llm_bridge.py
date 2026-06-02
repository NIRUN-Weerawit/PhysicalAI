"""
llm_bridge.py — LLM conversation loop with tool-calling.

Enhanced with rich TUI output: colored panels, reasoning display,
tool visualization, and prompt_toolkit input with history.
"""
import json, time, threading, sys, queue, os, re
from typing import Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich.markdown import Markdown

from orchestrator.robot_interface import ToolResult
from orchestrator.llm_tools import registry as tool_registry

# Rich console for pretty output
_console = Console()


class LLMBridge:
    """Manages the LLM conversation loop with rich TUI output."""

    def __init__(self, robot, tool_registry, model_name: str = None,
                 api_base: str = None, api_key: str = None,
                 telegram_gateway=None, dashboard_push=None):
        self._robot = robot
        self._tools = tool_registry
        self._model = model_name or "deepseek-v4-flash:cloud"
        self._api_base = api_base or os.environ.get("OLLAMA_HOST", "")
        self._api_key = api_key or ""
        self._telegram = telegram_gateway
        self._dashboard_push = dashboard_push  # callable(role, text) for web chat
        self._last_token_count = 0

        # ── Interrupt support ──
        self._processing = False      # True while LLM chain is executing
        self._interrupt = False       # Set to True to abort current chain

        # Conversation history
        self._messages = []
        self._max_history = 40

        # User command queue
        self._cmd_queue = queue.Queue()
        self._running = True
        self._places = {}

        # Load system prompt
        self._system_prompt = self._tools.generate_system_prompt()

    # ── Public API ──────────────────────────────────────────────────────────

    def inject_message(self, source: str, text: str):
        self._cmd_queue.put((source, text))

    def say(self, message: str):
        """Output from robot — rich panel + Telegram + dashboard chat."""
        _console.print(Panel(
            Text(message, style="bold green"),
            title="🤖 Robot",
            border_style="green",
            box=box.ROUNDED,
        ))
        if self._telegram:
            self._telegram.send_message(message)
        # Push to dashboard chat
        if self._dashboard_push:
            self._dashboard_push("robot", message)

    def confirm(self, question: str) -> bool:
        """Ask user yes/no with styled prompt."""
        if self._telegram:
            self._telegram.send_message(f"⚠️  {question} (reply y/n)")
        _console.print(Panel(
            Text(f"{question} (y/n)", style="bold yellow"),
            title="⚠️  Confirmation",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        try:
            ans = sys.stdin.readline().strip().lower()
            return ans in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def run(self):
        """Main conversation loop with TUI."""
        # Intro banner
        _console.rule("[bold blue]PhysicalAI — LLM Orchestrator[/bold blue]")
        _console.print()
        _console.print(f"  Model: [cyan]{self._model}[/cyan]")
        _console.print(f"  Tools: [cyan]{len(self._tools.all())}[/cyan] registered")
        _console.print(f"  Telegram: [cyan]{'enabled' if self._telegram else 'disabled'}[/cyan]")
        _console.print("  Type your commands. [dim]Ctrl+C to exit.[/dim]")
        _console.print()
        _console.rule()

        # System prompt
        self._messages.append({
            "role": "system",
            "content": self._system_prompt,
        })

        # Start CLI input thread
        self._cli_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._cli_thread.start()

        # Start status bar thread (periodic compact status updates)
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

        while self._running:
            try:
                source, text = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                # Check for interrupt requests (non-blocking)
                if self._interrupt and not self._processing:
                    self._interrupt = False
                continue
            if not text:
                continue

            # If currently processing, interrupt the chain
            if self._processing:
                self._interrupt = True
                # Drain pending interrupt requests
                while True:
                    try:
                        source, text = self._cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                _console.print("[bold yellow]⚡ Interrupted — processing new command.[/bold yellow]")
                if self._dashboard_push:
                    self._dashboard_push("system", "⚡ Interrupted — processing new command")
                # Wait for current processing to stop
                while self._processing:
                    time.sleep(0.1)

            # Echo user input
            _console.print(f"\n[bold cyan]>>>[/bold cyan] {text}")

            self._messages.append({
                "role": "user",
                "content": f"[{source}] {text}",
            })
            self._trim_history()
            self._processing = True
            self._interrupt = False
            # Run process_loop in background thread so main loop can handle interrupts
            proc_t = threading.Thread(target=self._process_loop, daemon=True)
            proc_t.start()
            # Wait for it to finish or for interrupt to clear
            while proc_t.is_alive():
                if self._interrupt:
                    time.sleep(0.1)
                proc_t.join(timeout=0.2)
            self._processing = False
            # Signal dashboard that processing is done
            try:
                from orchestrator.dashboard_server import set_processing
                set_processing(False)
            except Exception:
                pass
            if self._interrupt:
                self._interrupt = False

        _console.print("\n[bold red]Orchestrator stopped.[/bold red]")

    # ── Internal processing loop ────────────────────────────────────────────

    def _process_loop(self, max_turns: int = 8):
        for turn in range(max_turns):
            # Check for interrupt
            if self._interrupt:
                # Cancel active navigation goal
                robot = getattr(self, '_robot', None)
                if robot and getattr(robot, '_goal_active', False):
                    robot.stop()
                _console.print("[bold yellow]⚡ Processing interrupted.[/bold yellow]")
                if self._dashboard_push:
                    self._dashboard_push("system", "⚡ Processing interrupted.")
                return

            response_text = self._call_llm()
            if response_text is None:
                err_msg = "Couldn't reach the LLM. Check your config and connection."
                _console.print(Panel(
                    Text(err_msg, style="bold red"),
                    title="❌ LLM Error",
                    border_style="red",
                ))
                if self._dashboard_push:
                    self._dashboard_push("system", f"❌ LLM Error: {err_msg}")
                return

            self._messages.append({"role": "assistant", "content": response_text})

            # ── Extract and show thinking/reasoning ──
            thinking = self._extract_thinking(response_text)
            clean = self._strip_thinking(response_text)

            if thinking:
                _console.print(Panel(
                    Text(thinking, style="dim italic"),
                    title="💭 Thinking",
                    border_style="bright_black",
                    box=box.ROUNDED,
                ))
                if self._dashboard_push:
                    self._dashboard_push("thinking", f"{thinking[:500]}")

            # ── Parse JSON response ──
            # Try full parse first, then extract embedded JSON
            parsed = None
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                # Check for embedded JSON: text + {"tool": ...}
                import re as _re
                json_match = _re.search(r'\{\s*"(?:tool|reply|confirm)"', clean)
                if json_match:
                    # Find the matching closing brace
                    start = json_match.start()
                    depth = 0
                    for i in range(start, len(clean)):
                        if clean[i] == '{':
                            depth += 1
                        elif clean[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(clean[start:i+1])
                                    # Display the text that preceded the tool call
                                    preamble = clean[:start].strip()
                                    if preamble:
                                        _console.print(Panel(
                                            Markdown(preamble),
                                            title="💬 Response",
                                            border_style="blue",
                                        ))
                                        if self._dashboard_push:
                                            self._dashboard_push("robot", preamble[:300])
                                except json.JSONDecodeError:
                                    pass
                                break

            if parsed is None:
                # Not JSON at all — show as plain markdown response
                _console.print(Panel(
                    Markdown(clean),
                    title="💬 Response",
                    border_style="blue",
                ))
                if self._dashboard_push:
                    self._dashboard_push("robot", clean[:500])
                return

            if "reply" in parsed:
                _console.print(Panel(
                    Markdown(parsed["reply"]),
                    title="💬 Response",
                    border_style="blue",
                ))
                if self._dashboard_push:
                    self._dashboard_push("robot", parsed["reply"][:500])
                return

            if "confirm" in parsed:
                ok = self.confirm(parsed["confirm"])
                self._messages.append({
                    "role": "user",
                    "content": f"[system] User confirmed: {ok}",
                })
                continue

            if "tool" in parsed:
                tool_name = parsed["tool"]
                args = parsed.get("args", {})
                args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""

                # Create visual panel before execution
                _console.print(Panel(
                    Text(f"Calling: {tool_name}({args_str})", style="bold yellow"),
                    title="🔧 Tool Call",
                    border_style="yellow",
                    box=box.SIMPLE,
                ))
                if self._dashboard_push:
                    self._dashboard_push("system", f"🔧 Calling: {tool_name}({args_str})")

                if tool_name == "create_skill":
                    result = self._execute_create_skill(args)
                elif tool_name == "unregister_skill":
                    result = self._execute_unregister_skill(args)
                else:
                    result = self._execute_tool(tool_name, args)

                # Show result with status color
                if result.success:
                    _console.print(Panel(
                        Text(result.message[:300], style="green"),
                        title=f"✅ {tool_name}",
                        border_style="green",
                        box=box.SIMPLE,
                    ))
                    if self._dashboard_push:
                        self._dashboard_push("tool", f"{tool_name}: {result.message[:200]}")
                else:
                    _console.print(Panel(
                        Text(f"❌ {result.message[:300]}", style="red"),
                        title=f"⛔ {tool_name}",
                        border_style="red",
                        box=box.SIMPLE,
                    ))
                    if self._dashboard_push:
                        self._dashboard_push("tool", f"FAILED {tool_name}: {result.message[:200]}")

                self._messages.append({
                    "role": "user",
                    "content": f"[tool_result] {tool_name} returned: {result.message}",
                })
                continue

            # Unknown format
            _console.print(f"[dim]{clean[:500]}[/dim]")
            return

        _console.print("[bold yellow]I've taken too many actions. What should I do next?[/bold yellow]")

    # ── Thinking extraction ────────────────────────────────────────────────

    def _extract_thinking(self, text: str) -> str:
        """Extract reasoning/thinking from common formats."""
        # Format 1: <thinking>...</thinking>
        match = re.search(r'<thinking>\s*(.*?)\s*</thinking>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Format 2:  ... 
        match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Format 3:  ... 
        match = re.search(r'<Thought>\s*(.*?)\s*</Thought>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Format 4: # ... # reasoning blocks
        match = re.search(r'#+\s*Reason(?:ing)?\s*#+\s*(.*?)(?:#+\s*|\n\s*\n)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Format 5: // ... //  (code-style comments as reasoning)
        lines = text.split('\n')
        comment_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') and not stripped.startswith('// '):
                break
            if stripped.startswith('// ') or stripped.startswith('//'):
                comment_lines.append(stripped.lstrip('/').strip())
            elif comment_lines and stripped:
                break
        if comment_lines:
            return '\n'.join(comment_lines)

        # Format 6: DeepSeek native —  ... 
        match = re.search(r'<Think>\s*(.*?)\s*</Think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Format 7: Ollama / proxy native  ... 
        match = re.search(r'<\|begin_of_thought\|>\s*(.*?)\s*<\|end_of_thought\|>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""

    def _strip_thinking(self, text: str) -> str:
        """Remove all known thinking/reasoning formats for JSON parsing."""
        clean = text
        clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<reasoning>.*?</reasoning>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<Thought>.*?</Thought>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<Think>.*?</Think>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'#+\s*Reason(?:ing)?\s*#+.*?(?=#+\s*|\n\s*\n|$)', '', clean, flags=re.DOTALL)
        lines = clean.split('\n')
        lines = [l for l in lines if not l.strip().startswith('//') or l.strip() == '//']
        clean = '\n'.join(lines).strip()
        return clean

    # ── LLM calls ──────────────────────────────────────────────────────────

    def _call_llm(self) -> Optional[str]:
        if not self._api_key:
            return self._call_ollama()
        if self._api_base and ("localhost" in self._api_base or "11434" in self._api_base or "ollama" in self._api_base.lower()):
            return self._call_ollama()
        return self._call_openrouter()

    def _call_ollama(self) -> Optional[str]:
        host = self._api_base or "http://localhost:11434"
        url = f"{host.rstrip('/')}/v1/chat/completions"
        model = self._model.replace("ollama/", "")
        payload = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]}
                        for m in self._messages[-self._max_history:]],
            "stream": False,
            "temperature": 0.3,
        }
        try:
            _console.print("  [dim]Waiting for LLM...[/dim]")
            if self._dashboard_push:
                self._dashboard_push("system", "Thinking...")
                from orchestrator.dashboard_server import set_processing
                set_processing(True)
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            if usage:
                total = usage.get("total_tokens", 0) or usage.get("completion_tokens", 0)
                self._last_token_count = total
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "")
            if reasoning:
                content = f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{content}"
            return content
        except Exception as e:
            _console.print(Panel(
                Text(str(e), style="red"),
                title="[LLM Error]",
                border_style="red",
            ))
            return None

    def _call_openrouter(self) -> Optional[str]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": m["role"], "content": m["content"]}
                        for m in self._messages[-self._max_history:]],
            "temperature": 0.3,
        }
        try:
            _console.print("  [dim]Waiting for LLM...[/dim]")
            if self._dashboard_push:
                self._dashboard_push("system", "Waiting for LLM...")
            resp = requests.post(url, json=payload, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            _console.print(Panel(
                Text(str(e), style="red"),
                title="[LLM Error]",
                border_style="red",
            ))
            return None

    # ── Tool execution ─────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> ToolResult:
        td = self._tools.get(name)
        if td is None:
            return ToolResult(False, f"Tool '{name}' not found.", failure_type="internal_error")

        if name == "remember_place":
            pose = self._robot.get_pose()
            if not pose.success:
                return ToolResult(False, "Cannot save place: pose unknown.", failure_type="tf_timeout")
            return self._robot.remember_place(
                args.get("name", "unnamed"),
                pose.data.get("pose", (0, 0, 0)),
                self._places)

        if name == "go_to_place":
            return self._robot.go_to_place(args.get("name", ""), self._places)

        if name == "list_places":
            return self._robot.list_places(self._places)

        if name == "get_status":
            return self._robot.get_status(self._robot._object_db, self._robot._map)

        if name == "list_objects":
            return self._robot.list_objects()

        try:
            return td.handler(**args)
        except TypeError as e:
            return ToolResult(False, f"Tool '{name}' received bad args: {e}", failure_type="internal_error")
        except Exception as e:
            return ToolResult(False, f"Tool '{name}' execution failed: {e}", failure_type="internal_error")

    def _execute_create_skill(self, args: dict) -> ToolResult:
        name = args.get("name", "").strip()
        description = args.get("description", "")
        code = args.get("code", "")
        if not name or not code:
            return ToolResult(False, "create_skill requires 'name', 'description', and 'code'.", failure_type="internal_error")
        from orchestrator.llm_tools import PROTECTED_TOOLS
        if name in PROTECTED_TOOLS:
            return ToolResult(False, f"Cannot override protected tool: {name}", failure_type="internal_error")
        import os
        skill_dir = "/tmp/physicalai_skills"
        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(skill_dir, f"skill_{name}.py")
        with open(skill_path, "w") as f:
            f.write(code)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"skill_{name}", skill_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, name, None)
            if fn is None:
                return ToolResult(False, f"Skill file created but function '{name}' not found.")
            self._tools.register(name, description, fn)
            return ToolResult(True, f"Skill '{name}' created and registered. Use confirm() before first execution.")
        except Exception as e:
            return ToolResult(False, f"Failed to register skill '{name}': {e}", failure_type="internal_error")

    def _execute_unregister_skill(self, args: dict) -> ToolResult:
        name = args.get("name", "")
        if self._tools.unregister(name):
            return ToolResult(True, f"Skill '{name}' unregistered.")
        return ToolResult(False, f"Cannot unregister '{name}'. It may be protected.")

    def _trim_history(self):
        if len(self._messages) > self._max_history:
            system = [m for m in self._messages if m["role"] == "system"]
            others = [m for m in self._messages if m["role"] != "system"]
            self._messages = system + others[-(self._max_history - len(system)):]

    def stop(self):
        self._running = False

    # ── Status bar ──────────────────────────────────────────────────────────

    def _status_loop(self):
        """Periodic compact status bar at the bottom of the TUI."""
        while self._running:
            try:
                pose = self._robot.get_pose()
                if pose.success:
                    p = pose.data.get("pose", (0, 0, 0))
                    px, py, theta = p[0], p[1], p[2]
                else:
                    px, py, theta = 0, 0, 0

                db = self._robot._object_db
                obj_count = len(db.get_all()) if db else 0
                nav = "nav" if self._robot._goal_active else "idle"
                tool_count = len(self._tools.all())
                model_short = self._model.split("/")[-1] if "/" in self._model else self._model

                bar = (f" 📍 ({px:5.1f}, {py:5.1f})  "
                       f"🔍 {obj_count} obj  "
                       f"⚙ {tool_count} tools  "
                       f"🚗 {nav}  "
                       f"🤖 {model_short}")

                _console.print(bar, style="dim", markup=True, end="\r")
                time.sleep(2)
            except Exception:
                time.sleep(3)

    def _stdin_loop(self):
        """Background thread: read stdin and push into the command queue."""
        while self._running:
            try:
                text = sys.stdin.readline()
                if not text:
                    time.sleep(0.1)
                    continue
                text = text.strip()
                if text:
                    self._cmd_queue.put(("[cli]", text))
            except (EOFError, OSError):
                time.sleep(0.5)
