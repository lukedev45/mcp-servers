# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from pathlib import Path

import pytest
from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
)


# Get the path to the tests directory
TESTS_DIR = Path(__file__).parent.parent


def _validate_synthesis_result(result: dict) -> None:
    """Helper to validate synthesis result structure and values."""
    assert result["status"] == "success"
    assert isinstance(result["circuit_qpy"], str)
    assert len(result["circuit_qpy"]) > 0  # Non-empty QPY

    # Validate original_circuit metrics
    orig = result["original_circuit"]
    assert isinstance(orig, dict)
    assert "num_qubits" in orig and isinstance(orig["num_qubits"], int)
    assert "depth" in orig and isinstance(orig["depth"], int)
    assert "size" in orig and isinstance(orig["size"], int)
    assert "two_qubit_gates" in orig and isinstance(orig["two_qubit_gates"], int)
    assert orig["num_qubits"] > 0

    # Validate optimized_circuit metrics
    opt = result["optimized_circuit"]
    assert isinstance(opt, dict)
    assert "num_qubits" in opt and isinstance(opt["num_qubits"], int)
    assert "depth" in opt and isinstance(opt["depth"], int)
    assert "size" in opt and isinstance(opt["size"], int)
    assert "two_qubit_gates" in opt and isinstance(opt["two_qubit_gates"], int)
    # Qubit count should be preserved
    assert opt["num_qubits"] == orig["num_qubits"]

    # Validate improvements
    imp = result["improvements"]
    assert isinstance(imp, dict)
    assert "depth_reduction" in imp and isinstance(imp["depth_reduction"], int)
    assert "two_qubit_gate_reduction" in imp and isinstance(imp["two_qubit_gate_reduction"], int)
    # Verify improvement calculation is correct
    assert imp["depth_reduction"] == orig["depth"] - opt["depth"]
    assert imp["two_qubit_gate_reduction"] == orig["two_qubit_gates"] - opt["two_qubit_gates"]


class TestAIRoutingSync:
    """Test AI Routing sync method"""

    @pytest.mark.integration
    def test_ai_routing_sync_success(self, backend_name):
        """
        Successful test AI routing sync tool.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = ai_routing.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "success"

    @pytest.mark.integration
    def test_ai_routing_sync_failure_backend_name(
        self,
    ):
        """
        Failed test AI routing sync tool. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_routing.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_routing_sync_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI routing sync tool. Here we simulate wrong input QASM string.
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = ai_routing.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAICliffordSync:
    """Test AI Clifford synthesis sync method"""

    @pytest.mark.integration
    def test_ai_clifford_sync_success(self, backend_name):
        """
        Successful test AI Clifford synthesis sync tool.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = ai_clifford_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        _validate_synthesis_result(result)

    @pytest.mark.integration
    def test_ai_clifford_sync_failure_backend_name(self):
        """
        Failed test AI Clifford synthesis sync tool. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_clifford_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_clifford_sync_failure_wrong_qasm(self, backend_name):
        """
        Failed test AI Clifford synthesis sync tool. Here we simulate wrong qasm string
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = ai_clifford_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAILinearFunctionSync:
    """Test AI Linear Function synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_linear_function_sync_success(self, backend_name):
        """
        Successful test AI Linear Function synthesis sync tool.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = ai_linear_function_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        _validate_synthesis_result(result)

    @pytest.mark.integration
    def test_ai_linear_function_sync_failure_backend_name(self):
        """
        Failed test AI Linear Function synthesis sync tool. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_linear_function_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_linear_function_sync_failure_wrong_qasm(self, backend_name):
        """
        Failed test AI Linear Function synthesis sync tool. Here we simulate wrong qasm string
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = ai_linear_function_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAIPermutationSync:
    """Test AI Permutation synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_permutation_sync_success(self, backend_name):
        """
        Successful test AI Permutation synthesis sync tool.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = ai_permutation_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        _validate_synthesis_result(result)

    @pytest.mark.integration
    def test_ai_permutation_sync_failure_backend_name(self):
        """
        Failed test AI Permutation synthesis sync tool. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_permutation_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_permutation_sync_failure_wrong_qasm(self, backend_name):
        """
        Failed test AI Permutation synthesis sync tool. Here we simulate wrong qasm string
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = ai_permutation_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAIPauliNetworkSync:
    """Test AI Pauli Network synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_pauli_network_sync_success(self, backend_name):
        """
        Successful test AI Pauli Network synthesis sync tool.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = ai_pauli_network_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        _validate_synthesis_result(result)

    @pytest.mark.integration
    def test_ai_pauli_network_sync_failure_backend_name(self):
        """
        Failed test AI Pauli Network synthesis sync tool. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_pauli_network_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_pauli_network_sync_failure_wrong_qasm(self, backend_name):
        """
        Failed test AI Pauli Network synthesis sync tool. Here we simulate wrong qasm string
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = ai_pauli_network_synthesis.sync(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
