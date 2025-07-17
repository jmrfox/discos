#!/usr/bin/env python3
"""
VS Code Performance Debugging Script
Run this to help identify what's causing VS Code freezes.
"""

import subprocess
import time
import json


def get_vscode_processes():
    """Get VS Code process information."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Code.exe", "/FO", "CSV"], capture_output=True, text=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error getting processes: {e}")
        return ""


def disable_problematic_extensions():
    """Disable extensions that commonly cause freezes."""
    problematic_extensions = [
        "ms-python.vscode-pylance",  # Heavy Python analysis
        "eamodio.gitlens",  # Heavy Git operations
        "kaih2o.python-resource-monitor",  # Resource monitoring
        "atishay-jain.all-autocomplete",  # Heavy autocomplete
        "perpetualhelp.python-line-profiler",  # Line profiling
        "visualstudioexptteam.vscodeintellicode",  # IntelliCode
        "ms-toolsai.datawrangler",  # Data processing
        "codestream.codestream",  # CodeStream analysis
    ]

    print("🔧 Commands to disable problematic extensions:")
    print("Copy and run these commands one by one in PowerShell:\n")

    for ext in problematic_extensions:
        print(f"code --disable-extension {ext}")

    print("\n" + "=" * 60)
    print("After disabling extensions, restart VS Code and test.")
    print("If freezing stops, re-enable extensions one by one to find the culprit.")
    print("=" * 60)


def performance_tips():
    """Print performance optimization tips."""
    print("\n🚀 Additional Performance Tips:")
    print("1. Close unused tabs and windows")
    print("2. Disable preview mode for files")
    print("3. Limit Python analysis to open files only")
    print("4. Exclude large directories from file watching")
    print("5. Turn off real-time linting")
    print("6. Disable semantic highlighting")
    print("7. Reduce extension auto-updates")


if __name__ == "__main__":
    print("🔍 VS Code Performance Debugger")
    print("=" * 40)

    print("\n📊 Current VS Code Processes:")
    processes = get_vscode_processes()
    print(processes)

    disable_problematic_extensions()
    performance_tips()

    print("\n⚡ Settings already applied to your workspace:")
    print("- Disabled Python indexing")
    print("- Reduced file watching")
    print("- Disabled semantic highlighting")
    print("- Turned off autocomplete suggestions")
    print("- Disabled linting")
