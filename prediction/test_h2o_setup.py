"""
Test H2O setup and basic functionality
Quick verification script to ensure H2O can initialize properly in HPC environment
"""

import os
import sys
import multiprocessing

def test_h2o_import():
    """Test if H2O can be imported."""
    try:
        import h2o
        print("✓ H2O import successful")
        print(f"  H2O version: {h2o.__version__}")
        return True
    except ImportError as e:
        print(f"✗ H2O import failed: {e}")
        return False

def test_h2o_initialization():
    """Test H2O cluster initialization."""
    try:
        import h2o

        # Get environment settings
        num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
        available_memory = os.environ.get('SLURM_MEM_PER_NODE', '64G')

        if available_memory.endswith('G'):
            memory_gb = int(available_memory[:-1])
            h2o_memory = f"{max(4, memory_gb - 8)}G"
        else:
            h2o_memory = "32G"

        print(f"Attempting to initialize H2O with:")
        print(f"  CPUs: {num_cpus}")
        print(f"  Memory: {h2o_memory}")

        # Initialize H2O
        h2o.init(
            nthreads=num_cpus,
            max_mem_size=h2o_memory,
            strict_version_check=False,
            ignore_config=True
        )

        print("✓ H2O cluster initialization successful")
        print("Cluster info:")
        print(h2o.cluster().show())

        # Shutdown cluster
        h2o.cluster().shutdown()
        print("✓ H2O cluster shutdown successful")
        return True

    except Exception as e:
        print(f"✗ H2O initialization failed: {e}")
        return False

def test_basic_h2o_operations():
    """Test basic H2O operations."""
    try:
        import h2o
        import pandas as pd
        import numpy as np

        # Initialize H2O
        h2o.init(nthreads=2, max_mem_size="4G", strict_version_check=False, ignore_config=True)

        # Create test dataset
        test_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randint(0, 10, 1000),
            'target': np.random.randn(1000)
        })

        # Convert to H2OFrame
        h2o_frame = h2o.H2OFrame(test_data)
        print(f"✓ H2OFrame creation successful: {h2o_frame.shape}")

        # Test basic operations
        print(f"  Frame summary:")
        print(h2o_frame.describe())

        # Test train/test split
        train, test = h2o_frame.split_frame(ratios=[0.8], seed=42)
        print(f"✓ Train/test split successful: train={train.nrows}, test={test.nrows}")

        # Test a simple model
        from h2o.estimators import H2OGradientBoostingEstimator

        gbm = H2OGradientBoostingEstimator(
            ntrees=10,
            max_depth=3,
            seed=42
        )

        gbm.train(
            x=['feature1', 'feature2', 'feature3'],
            y='target',
            training_frame=train,
            validation_frame=test
        )

        print("✓ Basic GBM model training successful")

        # Test predictions
        predictions = gbm.predict(test)
        print(f"✓ Predictions successful: {predictions.nrows} predictions made")

        # Cleanup
        h2o.cluster().shutdown()
        print("✓ All basic H2O operations successful")
        return True

    except Exception as e:
        print(f"✗ Basic H2O operations failed: {e}")
        try:
            h2o.cluster().shutdown()
        except:
            pass
        return False

def test_environment_setup():
    """Test environment configuration."""
    print("Environment Configuration:")
    print(f"  Python version: {sys.version}")
    print(f"  SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}")
    print(f"  SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE', 'Not set')}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print(f"  _JAVA_OPTIONS: {os.environ.get('_JAVA_OPTIONS', 'Not set')}")
    print(f"  Available CPUs: {multiprocessing.cpu_count()}")

def main():
    """Run all H2O tests."""
    print("=" * 60)
    print("H2O.ai Setup Verification for HPC Environment")
    print("=" * 60)

    # Test environment
    test_environment_setup()
    print()

    # Test H2O import
    if not test_h2o_import():
        print("Cannot proceed without H2O import")
        return False
    print()

    # Test H2O initialization
    if not test_h2o_initialization():
        print("H2O initialization failed - check memory/CPU settings")
        return False
    print()

    # Test basic operations
    if not test_basic_h2o_operations():
        print("Basic H2O operations failed")
        return False

    print("=" * 60)
    print("✓ ALL H2O TESTS PASSED - Ready for production training!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)