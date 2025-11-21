"""CLI utilities (R ricu utils-cli.R).

Functions for CLI interaction, progress reporting, and formatted output.
"""

import sys
from typing import Any, Callable, List, Optional
from contextlib import contextmanager

# ============================================================================
# Interactive check
# ============================================================================

def is_interactive() -> bool:
    """Check if running in interactive mode (R ricu is_interactive).
    
    Returns:
        True if running interactively
    """
    return sys.__stdin__.isatty()

# ============================================================================
# Progress reporting
# ============================================================================

class ProgressBar:
    """Simple progress bar (R ricu progress_bar)."""
    
    def __init__(self, total: int, msg: Optional[str] = None, what: bool = True):
        """Initialize progress bar.
        
        Args:
            total: Total number of items
            msg: Message to display
            what: Whether to show item names
        """
        self.total = total
        self.current = 0
        self.msg = msg
        self.what = what
        self.finished = False
        self.messages = []
        self.header = None
        self.token = None
        
        if msg:
            print(f"{'=' * 60}")
            print(f"{msg}")
            print(f"{'=' * 60}")
    
    def tick(self, n: int = 1, token: Optional[str] = None):
        """Advance progress bar.
        
        Args:
            n: Number of items to advance
            token: Item name to display
        """
        self.current += n
        
        if token:
            self.token = token
        
        if is_interactive() and self.what:
            pct = int((self.current / self.total) * 100)
            bar_len = 40
            filled = int((self.current / self.total) * bar_len)
            bar = '=' * filled + '-' * (bar_len - filled)
            
            display_token = self.token[:15] if self.token else ""
            sys.stdout.write(f'\r{display_token:<15} [{bar}] {pct}%')
            sys.stdout.flush()
            
            if self.current >= self.total:
                sys.stdout.write('\n')
                self.finished = True
    
    def update(self, ratio: float):
        """Update to specific ratio.
        
        Args:
            ratio: Completion ratio (0-1)
        """
        target = int(self.total * ratio)
        if target > self.current:
            self.tick(target - self.current)

def progress_init(length: Optional[int] = None, msg: str = "loading",
                 what: bool = True) -> Optional[ProgressBar]:
    """Initialize progress bar (R ricu progress_init).
    
    Args:
        length: Total number of items
        msg: Message to display
        what: Whether to show item names
        
    Returns:
        ProgressBar or None if not interactive
    """
    if is_interactive() and length and length > 1:
        return ProgressBar(total=length, msg=msg, what=what)
    else:
        if msg:
            print(f"{'=' * 60}")
            print(f"{msg}")
            print(f"{'=' * 60}")
        return None

def progress_tick(info: Optional[str] = None, progress_bar: Optional[ProgressBar] = None,
                 length: int = 1):
    """Tick progress bar (R ricu progress_tick).
    
    Args:
        info: Item information
        progress_bar: ProgressBar instance
        length: Number of items to advance
    """
    if progress_bar is False:
        return
    
    if progress_bar is None:
        if info:
            print(f"  • {info}")
        return
    
    if info:
        # Truncate long names
        if len(info) > 15:
            token = info[:12] + "..."
        else:
            token = info
        
        progress_bar.tick(n=length, token=token)
    else:
        progress_bar.tick(n=length)

@contextmanager
def with_progress(progress_bar: Optional[ProgressBar] = None):
    """Context manager for progress reporting (R ricu with_progress).
    
    Args:
        progress_bar: ProgressBar instance
        
    Yields:
        Progress bar
    """
    try:
        yield progress_bar
    finally:
        if progress_bar and not progress_bar.finished:
            progress_bar.update(1.0)
        
        if progress_bar is not False:
            print(f"{'=' * 60}")

# ============================================================================
# Message formatting
# ============================================================================

def msg_ricu(msg: str, type: str = "info", indent: int = 0, exdent: int = 0):
    """Print formatted message (R ricu msg_ricu).
    
    Args:
        msg: Message to print
        type: Message type (info, warning, error, progress_header, progress_body)
        indent: Left indent
        exdent: Extra indent for wrapped lines
    """
    prefix = " " * indent
    
    if type == "warning":
        print(f"{prefix}⚠ {msg}")
    elif type == "error":
        print(f"{prefix}✗ {msg}")
    elif type == "progress_header":
        print(f"{prefix}• {msg}")
    elif type == "progress_body":
        print(f"{prefix}  ○ {msg}")
    else:
        print(f"{prefix}{msg}")

def warn_ricu(msg: str, **kwargs):
    """Print warning message (R ricu warn_ricu).
    
    Args:
        msg: Warning message
        **kwargs: Additional arguments for msg_ricu
    """
    msg_ricu(msg, type="warning", **kwargs)

def stop_ricu(msg: str, **kwargs):
    """Print error message and raise exception (R ricu stop_ricu).
    
    Args:
        msg: Error message
        **kwargs: Additional arguments for msg_ricu
        
    Raises:
        RuntimeError
    """
    msg_ricu(msg, type="error", **kwargs)
    raise RuntimeError(msg)

def msg_progress(msg: str):
    """Print progress message (R ricu msg_progress).
    
    Args:
        msg: Message to print
    """
    msg_ricu(msg, type="progress_body", indent=2, exdent=4)

# ============================================================================
# Formatting utilities
# ============================================================================

def fmt_msg(msg: str, indent: int = 0, exdent: int = 0, width: int = 80) -> str:
    """Format message with wrapping (R ricu fmt_msg).
    
    Args:
        msg: Message to format
        indent: Left indent
        exdent: Extra indent for wrapped lines
        width: Maximum line width
        
    Returns:
        Formatted message
    """
    import textwrap
    
    lines = msg.split('\n')
    formatted_lines = []
    
    for line in lines:
        if len(line) <= width - indent:
            formatted_lines.append(' ' * indent + line)
        else:
            wrapper = textwrap.TextWrapper(
                width=width,
                initial_indent=' ' * indent,
                subsequent_indent=' ' * (indent + exdent)
            )
            formatted_lines.extend(wrapper.wrap(line))
    
    return '\n'.join(formatted_lines)

def bullet(text: str, level: int = 1) -> str:
    """Create bulleted text (R ricu bullet).
    
    Args:
        text: Text to bullet
        level: Bullet level (1, 2, or 3)
        
    Returns:
        Bulleted string
    """
    bullets = {1: "•", 2: "○", 3: "-"}
    bullet_char = bullets.get(level, "•")
    return f"{bullet_char} {text}"

def big_mark(x: int, sep: str = ",") -> str:
    """Format number with thousands separator (R ricu big_mark).
    
    Args:
        x: Number to format
        sep: Separator character
        
    Returns:
        Formatted number
    """
    return f"{x:,}".replace(",", sep)

def quote_bt(x: str) -> str:
    """Quote string with backticks (R ricu quote_bt).
    
    Args:
        x: String to quote
        
    Returns:
        Quoted string
    """
    return f"`{x}`"

def enbraket(x: str) -> str:
    """Surround with brackets (R ricu enbraket).
    
    Args:
        x: String to bracket
        
    Returns:
        Bracketed string
    """
    return f"[{x}]"

def concat(*args, sep: str = ", ") -> str:
    """Concatenate strings (R ricu concat).
    
    Args:
        *args: Strings to concatenate
        sep: Separator
        
    Returns:
        Concatenated string
    """
    return sep.join(str(arg) for arg in args)

def prcnt(x: float, tot: Optional[float] = None, digits: int = 2) -> str:
    """Format as percentage (R ricu prcnt).
    
    Args:
        x: Value
        tot: Total (if None, assumes x is already a percentage)
        digits: Decimal places
        
    Returns:
        Formatted percentage
    """
    if tot is not None:
        pct = (x / tot) * 100
    else:
        pct = x
    
    return f"{pct:.{digits}f}%"

# ============================================================================
# User interaction
# ============================================================================

def ask_yes_no(question: str, default: Optional[bool] = None) -> bool:
    """Ask yes/no question (R ricu ask_yes_no).
    
    Args:
        question: Question to ask
        default: Default answer
        
    Returns:
        True if yes, False if no
    """
    if not is_interactive():
        if default is None:
            raise RuntimeError("Non-interactive session requires default answer")
        return default
    
    if default is True:
        prompt = f"{question} [Y/n]: "
    elif default is False:
        prompt = f"{question} [y/N]: "
    else:
        prompt = f"{question} [y/n]: "
    
    while True:
        response = input(prompt).strip().lower()
        
        if not response:
            if default is not None:
                return default
        elif response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        
        print("Please answer 'y' or 'n'")

# ============================================================================
# CLI rule
# ============================================================================

def cli_rule(msg: str = "", width: int = 60):
    """Print CLI rule/separator (R ricu cli::cli_rule).
    
    Args:
        msg: Optional message
        width: Rule width
    """
    if msg:
        # Center message with rule on both sides
        msg_len = len(msg)
        side_len = (width - msg_len - 2) // 2
        rule = "=" * side_len + f" {msg} " + "=" * side_len
        # Adjust for odd widths
        if len(rule) < width:
            rule += "="
        print(rule)
    else:
        print("=" * width)

# ============================================================================
# Color support (optional rich integration)
# ============================================================================

class ColorText:
    """Simple color text wrapper (optional rich integration).
    
    Provides colored text output if rich library is available.
    Falls back to plain text otherwise.
    """
    
    def __init__(self):
        """Initialize color support."""
        try:
            from rich.console import Console
            self.console = Console()
            self.has_color = True
        except ImportError:
            self.console = None
            self.has_color = False
    
    def print(self, text: str, style: Optional[str] = None):
        """Print colored text.
        
        Args:
            text: Text to print
            style: Rich style string (e.g., 'bold red', 'green', 'yellow')
        """
        if self.has_color and style:
            self.console.print(text, style=style)
        else:
            print(text)
    
    def info(self, text: str):
        """Print info message (blue)."""
        self.print(f"ℹ {text}", style="blue")
    
    def success(self, text: str):
        """Print success message (green)."""
        self.print(f"✓ {text}", style="green")
    
    def warning(self, text: str):
        """Print warning message (yellow)."""
        self.print(f"⚠ {text}", style="yellow")
    
    def error(self, text: str):
        """Print error message (red)."""
        self.print(f"✗ {text}", style="bold red")
    
    def heading(self, text: str):
        """Print heading (bold)."""
        self.print(text, style="bold")

# Global color text instance
color_text = ColorText()

def print_info(msg: str):
    """Print info message with color."""
    color_text.info(msg)

def print_success(msg: str):
    """Print success message with color."""
    color_text.success(msg)

def print_warning(msg: str):
    """Print warning message with color."""
    color_text.warning(msg)

def print_error(msg: str):
    """Print error message with color."""
    color_text.error(msg)

def print_heading(msg: str):
    """Print heading with style."""
    color_text.heading(msg)

