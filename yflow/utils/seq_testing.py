import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from ..core.model import Model
from ..core.device import Device
from ..layers.rnn import YQuence
from ..layers.lstm import YSTM


class SequenceTester:
    """Helper class for testing sequence handling, shape management, and GPU operations"""

    def __init__(self, device: str = 'cpu'):
        self.device = Device(device)

    def verify_shapes(self,
                      model: 'Model',
                      input_data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
        """Verify shape handling through the model with GPU support"""
        results = {}

        try:
            # Move input to correct device
            if isinstance(input_data, list):
                input_data = [self.device.to_device(x) for x in input_data]
            else:
                input_data = self.device.to_device(input_data)

            # Test forward pass shape handling
            output = model.predict(input_data)
            results['forward_pass'] = True
            results['output_shape'] = tuple(output.shape)
            results['output_device'] = 'gpu' if self.device.device_type == 'gpu' else 'cpu'

            # Get the RNN/LSTM layer
            sequence_layer = next(
                layer for layer in model.layers
                if isinstance(layer, (YQuence, YSTM))
            )

            # Check mask generation
            mask = sequence_layer.get_mask()
            results['has_mask'] = mask is not None
            if mask is not None:
                results['mask_shape'] = tuple(mask.shape)

            # Verify sequence lengths
            if isinstance(input_data, list):
                seq_lengths = [len(seq) for seq in input_data]
                results['sequence_lengths'] = seq_lengths

            # Memory usage if on GPU
            if self.device.device_type == 'gpu':
                try:
                    import cupy as cp
                    mem_info = cp.cuda.Device().mem_info
                    results['gpu_memory'] = {
                        'total': mem_info[0],
                        'free': mem_info[1],
                        'used': mem_info[0] - mem_info[1]
                    }
                except Exception as e:
                    results['gpu_memory_error'] = str(e)

        except Exception as e:
            results['error'] = str(e)
            results['forward_pass'] = False

        return results

    def generate_test_sequences(self,
                                num_sequences: int = 5,
                                min_length: int = 2,
                                max_length: int = 5,
                                num_features: int = 1,
                                seed: Optional[int] = None) -> List[Union[np.ndarray, 'cp.ndarray']]:
        """Generate test sequences of varying lengths with GPU support"""
        if seed is not None:
            np.random.seed(seed)
            if self.device.device_type == 'gpu':
                import cupy as cp
                cp.random.seed(seed)

        xp = self.device.xp
        sequences = []
        for _ in range(num_sequences):
            length = np.random.randint(min_length, max_length + 1)
            seq = xp.random.randn(length, num_features)
            sequences.append(seq)
        return sequences

    def run_shape_tests(self, model: 'Model',
                        batch_sizes: List[int] = [1, 32],
                        feature_dims: List[int] = [1, 10]) -> Dict[str, Any]:
        """Run comprehensive shape handling tests with GPU support"""
        results = {}

        for batch_size in batch_sizes:
            for feature_dim in feature_dims:
                # Test group name
                test_group = f"batch_{batch_size}_features_{feature_dim}"
                results[test_group] = {}

                # Test 1: Single fixed-length sequence
                fixed_seq = self.device.xp.random.randn(batch_size, 4, feature_dim)
                results[test_group]['fixed_length'] = self.verify_shapes(model, fixed_seq)

                # Test 2: Variable length sequences
                var_sequences = self.generate_test_sequences(
                    batch_size, 2, 5, feature_dim
                )
                results[test_group]['variable_length'] = self.verify_shapes(
                    model, var_sequences
                )

                # Test 3: Single timestep
                single_step = self.device.xp.random.randn(batch_size, 1, feature_dim)
                results[test_group]['single_step'] = self.verify_shapes(
                    model, single_step
                )

                # Test 4: Long sequence
                long_seq = self.device.xp.random.randn(batch_size, 100, feature_dim)
                results[test_group]['long_sequence'] = self.verify_shapes(
                    model, long_seq
                )

        return results

    @staticmethod
    def print_test_results(results: Dict[str, Any],
                           verbose: bool = True,
                           show_gpu_info: bool = True):
        """Print test results in a readable format"""
        print("\nShape Handling Test Results")
        print("=" * 50)

        for test_group, group_results in results.items():
            print(f"\nTest Group: {test_group}")
            print("-" * 30)

            for test_name, test_results in group_results.items():
                print(f"\n{test_name}:")
                if verbose:
                    for key, value in test_results.items():
                        if show_gpu_info or not key.startswith('gpu_'):
                            print(f"  {key}: {value}")
                else:
                    print(f"  Success: {test_results.get('forward_pass', False)}")
                    if 'error' in test_results:
                        print(f"  Error: {test_results['error']}")

    def stress_test(self,
                    model: 'Model',
                    max_batch_size: int = 1024,
                    max_seq_length: int = 1000,
                    max_features: int = 100) -> Dict[str, Any]:
        """Run stress tests to find performance limits"""
        results = {'limits': {}, 'errors': {}}

        try:
            # Test batch size scaling
            for batch_size in [2 ** i for i in range(int(np.log2(max_batch_size)))]:
                try:
                    data = self.device.xp.random.randn(batch_size, 10, 10)
                    _ = model.predict(data)
                    results['limits']['max_batch_size'] = batch_size
                except Exception as e:
                    results['errors']['batch_size'] = str(e)
                    break

            # Test sequence length scaling
            for seq_length in [2 ** i for i in range(int(np.log2(max_seq_length)))]:
                try:
                    data = self.device.xp.random.randn(32, seq_length, 10)
                    _ = model.predict(data)
                    results['limits']['max_seq_length'] = seq_length
                except Exception as e:
                    results['errors']['seq_length'] = str(e)
                    break

            # Test feature dimension scaling
            for feature_dim in [2 ** i for i in range(int(np.log2(max_features)))]:
                try:
                    data = self.device.xp.random.randn(32, 10, feature_dim)
                    _ = model.predict(data)
                    results['limits']['max_features'] = feature_dim
                except Exception as e:
                    results['errors']['features'] = str(e)
                    break

        except Exception as e:
            results['errors']['general'] = str(e)

        return results