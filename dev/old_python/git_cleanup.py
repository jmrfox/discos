#!/usr/bin/env python3
"""
Git Cleanup Script - Remove tracked files that should be ignored
This script safely removes files from git tracking while keeping them locally.
"""

import subprocess
import os


def run_git_command(cmd):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def clean_tracked_ignored_files():
    """Remove files from git that should be ignored."""

    print("🧹 Git Repository Cleanup")
    print("=" * 50)

    # Files and patterns to remove from tracking
    patterns_to_remove = ["__pycache__/", "*.pyc", "*.egg-info/", ".pytest_cache/", ".coverage", "htmlcov/"]

    print("📋 Files to remove from git tracking:")

    # First, let's see what will be removed
    for pattern in patterns_to_remove:
        success, stdout, stderr = run_git_command(f'git ls-files "{pattern}"')
        if success and stdout.strip():
            files = stdout.strip().split("\n")
            print(f"\n  {pattern}:")
            for file in files[:5]:  # Show first 5 files
                print(f"    - {file}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more files")

    print("\n" + "=" * 50)
    print("🚨 IMPORTANT: This will remove files from git but keep them locally!")
    print("=" * 50)

    # The safe way to do this
    commands = []

    # Method 1: Remove specific file types
    commands.extend(
        [
            "git rm -r --cached __pycache__/",
            "git rm -r --cached gencomo.egg-info/",
            "git rm --cached gencomo/__init__.pyc",
        ]
    )

    # Method 2: Remove all .pyc files
    commands.append("git rm --cached gencomo/**/*.pyc")
    commands.append("git rm --cached tests/**/*.pyc")

    print("Commands to run:")
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")

    return commands


def main():
    commands = clean_tracked_ignored_files()

    print("\n🔧 Next Steps:")
    print("1. Review the files listed above")
    print("2. Run the commands below to remove them from git")
    print("3. Commit the changes")
    print("4. Files will remain on your local filesystem")

    print("\n📝 Copy and paste these commands:")
    print("-" * 40)
    for cmd in commands:
        print(cmd)

    print("\n# Then commit the changes:")
    print("git add .")
    print('git commit -m "Remove tracked files that should be ignored"')

    print("\n✅ After this, .gitignore will properly ignore these file types!")


if __name__ == "__main__":
    main()
