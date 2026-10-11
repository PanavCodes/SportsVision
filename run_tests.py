#!/usr/bin/env python
"""
CourtVision / SportsVision Automated Test Runner.
Discovers and executes unit test suites across core spatial analytics components.
"""

import unittest
import sys
import time

def main():
    print("=" * 70)
    print(" CourtVision Automated Test Suite")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f" Tests Run:    {result.testsRun}")
    print(f" Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f" Failures:     {len(result.failures)}")
    print(f" Errors:       {len(result.errors)}")
    print(f" Elapsed Time: {elapsed:.2f}s")
    print("=" * 70)
    
    if not result.wasSuccessful():
        sys.exit(1)
    else:
        print("[✓] All CourtVision test suites passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
