---
description: Activate Python virtual environment (.venv) for the project
---

1. **Navigate to the project root**
   ```bash
   cd /home/missoum/Desktop/Projects/LLM-Agents
   ```

2. **Check if the virtual environment exists**
   ```bash
   if [ -d ".venv" ]; then
       echo "Virtual environment already exists."
   else
       echo "Creating virtual environment..."
       # // turbo
       python3 -m venv .venv
   fi
   ```

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```
   After activation, your shell prompt should change (e.g., `(venv)` prefix) and `which python` should point to the `.venv` directory.

4. **Verify activation**
   ```bash
   python -V   # should show the Python version inside .venv
   pip list    # should show packages installed in .venv
   ```

5. **Deactivate when done**
   ```bash
   deactivate
   ```

**Notes**:
- Use `python3` if your system defaults to Python 2.
- On Windows, activation commands differ (`.venv\Scripts\activate` for cmd, `\.venv\Scripts\Activate.ps1` for PowerShell).
- If you need to reinstall dependencies, run `pip install -r requirements.txt` after activation.
