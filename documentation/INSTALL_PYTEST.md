# Installing pytest for Testing

## Quick Install

The test runner requires `pytest` to be installed. Here's how to install it:

### Option 1: Install from requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

This will install all dependencies including pytest.

### Option 2: Install pytest only

```bash
pip install pytest pytest-cov
```

### Option 3: Install in development mode

```bash
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check if pytest is installed
pytest --version

# Should output something like:
# pytest 7.4.0
```

## Run Tests

Once pytest is installed, you can run the tests:

```bash
# Run all tests with results saved to file
python scripts/run_tests.py

# Results will be saved to: test_results.txt
```

## Troubleshooting

### Issue: "No module named pytest"

**Solution**: Install pytest using one of the methods above.

### Issue: Permission denied

**Solution**: Use `--user` flag:
```bash
pip install --user pytest pytest-cov
```

### Issue: Virtual environment

If using a virtual environment, make sure it's activated:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Then install pytest:
```bash
pip install pytest pytest-cov
```

## Next Steps

After installing pytest:

1. Run the test suite:
   ```bash
   python scripts/run_tests.py
   ```

2. Check the results in `test_results.txt`

3. If tests fail due to missing dependencies, install them:
   ```bash
   pip install -r requirements.txt
   ```

---

**Note**: The test results will automatically be saved to `test_results.txt` in the project root directory.
