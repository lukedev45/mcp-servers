from qiskit_ibm_transpiler_mcp_server.sync import (
    ai_routing_sync,
    ai_clifford_synthesis_sync,
    ai_linear_function_synthesis_sync,
    ai_pauli_network_synthesis_sync,
    ai_permutation_synthesis_sync,
    get_backend_service_sync,
    least_busy_backend_sync,
    list_backends_sync,
    get_backend_properties_sync,
    list_my_jobs_sync,
    cancel_job_sync,
    get_job_status_sync,
)

import pytest


class TestAIRoutingSync:
    """Test AIRouting sync tool."""

    def test_ai_routing_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI routing sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit_qasm",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = ai_routing_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["optimized_circuit_qasm"] == "optimized_circuit_qasm"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_routing_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "Error in loading QuantumCircuit from QASM3.0",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_routing_success",
                "QASM dumps failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_failure",
                "AIRouting failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_routing_fixture",
        ],
    )
    def test_ai_routing_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        ai_routing_fixture,
        dumps_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI routing sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_routing_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAICliffordSync:
    """Test AI Clifford synthesis sync tool."""

    def test_ai_clifford_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Clifford synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit_qasm",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = ai_clifford_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["optimized_circuit_qasm"] == "optimized_circuit_qasm"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_clifford_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "Error in loading QuantumCircuit from QASM3.0",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_clifford_synthesis_success",
                "QASM dumps failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_failure",
                "AI Clifford synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_clifford_synthesis_fixture",
        ],
    )
    def test_ai_clifford_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_clifford_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Clifford synthesis sync tool with existing backend, quantum circuit and PassManager.
        """
        result = ai_clifford_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAILinearFunctionSync:
    """Test AI Linear Function synthesis sync tool."""

    def test_ai_linear_function_sync_success(
        self, mocker, mock_circuit_qasm, mock_backend
    ):
        """
        Successful test AI Linear Function synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit_qasm",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = ai_linear_function_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["optimized_circuit_qasm"] == "optimized_circuit_qasm"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_linear_function_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "Error in loading QuantumCircuit from QASM3.0",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_linear_function_synthesis_success",
                "QASM dumps failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_failure",
                "AI Linear Function synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_linear_function_synthesis_fixture",
        ],
    )
    def test_ai_linear_function_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_linear_function_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Linear Function synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_linear_function_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPermutationSync:
    """Test AI Permutation synthesis sync tool."""

    def test_ai_permutation_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Permutation synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit_qasm",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = ai_permutation_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["optimized_circuit_qasm"] == "optimized_circuit_qasm"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_permutation_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "Error in loading QuantumCircuit from QASM3.0",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_permutation_synthesis_success",
                "QASM dumps failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_failure",
                "Permutation synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_permutation_synthesis_fixture",
        ],
    )
    def test_ai_permutation_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_permutation_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Permutation synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_permutation_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPauliNetworkSync:
    """Test AI Pauli Network synthesis sync tool."""

    def test_ai_pauli_network_sync_success(
        self, mocker, mock_circuit_qasm, mock_backend
    ):
        """
        Successful test AI Pauli Network synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit_qasm",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = ai_pauli_network_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["optimized_circuit_qasm"] == "optimized_circuit_qasm"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_pauli_networks_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "Error in loading QuantumCircuit from QASM3.0",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_pauli_network_synthesis_success",
                "QASM dumps failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_failure",
                "Pauli Networks synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_pauli_networks_synthesis_fixture",
        ],
    )
    def test_ai_pauli_networks_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_pauli_networks_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Pauli Network synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_pauli_network_synthesis_sync(
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestGetBackendSync:
    """Test get_backend function sync."""

    def test_ai_get_backend_sync_success(self, mocker, mock_backend):
        """
        Successful test get_backend_service sync tool.
        """
        mock_response = {
            "status": "success",
            "backend": "fake_backend",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = get_backend_service_sync(backend_name=mock_backend)
        assert result["status"] == "success"
        assert result["backend"] == "fake_backend"
        mocker_run_sync.assert_called_once()

    def test_ai_get_backend_sync_failure(self, mocker, mock_backend):
        """
        Failed test get_backend_service sync tool.
        """
        mock_response = {
            "status": "error",
            "message": "No backend 'fake_backend' available",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = get_backend_service_sync(backend_name=mock_backend)
        assert result["status"] == "error"
        assert result["message"] == "No backend 'fake_backend' available"
        mocker_run_sync.assert_called_once()


class TestLeastBusyBackendSync:
    """Test least_busy_backend sync function."""

    def test_least_busy_backend_sync_success(self, mocker):
        """
        Successful test least_busy_backend sync tool.
        """
        mock_response = {
            "status": "success",
            "backend_name": "fake_backend",
            "num_qubits": 127,
            "pending_jobs": 100,
            "operational": True,
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = least_busy_backend_sync()
        assert result["status"] == "success"
        assert result["backend_name"] == "fake_backend"
        assert result["num_qubits"] == 127
        assert result["pending_jobs"] == 100
        assert result["operational"] is True
        mocker_run_sync.assert_called_once()

    def test_least_busy_backend_sync_failure(self, mocker):
        """
        Failed test least_busy_backend sync tool
        """
        mock_response = {
            "status": "error",
            "message": "No operational quantum backends available",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async",
            return_value=mock_response,
        )
        result = least_busy_backend_sync()
        assert result["status"] == "error"
        assert result["message"] == "No operational quantum backends available"
        mocker_run_sync.assert_called_once()


class TestListBackendSync:
    """Test list_backend sync function."""

    def test_list_backends_sync_success(self, mocker):
        """Successful test backends listing with sync wrapper."""
        mock_response = {
            "status": "success",
            "backends": [
                {
                    "name": "ibm_brisbane",
                    "num_qubits": 133,
                    "simulator": False,
                    "operational": True,
                    "pending_jobs": 5,
                },
                {
                    "name": "ibm_kyoto",
                    "num_qubits": 127,
                    "simulator": False,
                    "operational": True,
                    "pending_jobs": 10,
                },
            ],
            "total_backends": 2,
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = list_backends_sync()

        assert result["status"] == "success"
        assert result["total_backends"] == 2
        assert len(result["backends"]) == 2
        mocker_run_sync.assert_called_once()

    def test_list_backends_sync_error(self, mocker):
        """Test error handling in sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to list backends: service not initialized",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = list_backends_sync()

        assert result["status"] == "error"
        mocker_run_sync.assert_called_once()


class TestGetBackendPropertiesSync:
    """Test get_backend_properties_sync function."""

    def test_get_backend_properties_sync_success(self, mocker):
        """Test successful backend properties retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 133,
            "simulator": False,
            "operational": True,
            "basis_gates": ["id", "rz", "sx", "x", "cx"],
            "max_shots": 100000,
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = get_backend_properties_sync("ibm_brisbane")

        assert result["status"] == "success"
        assert result["backend_name"] == "ibm_brisbane"
        assert result["num_qubits"] == 133

    def test_get_backend_properties_sync_failure(self, mocker):
        """Failed test backend properties retrieval with sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to get backend properties",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = get_backend_properties_sync("ibm_brisbane")

        assert result["status"] == "error"
        assert result["message"] == "Failed to get backend properties"


class TestListMyJobsSync:
    """Test list_my_jobs_sync function."""

    def test_list_my_jobs_sync_success(self, mocker):
        """Test successful jobs listing with sync wrapper."""
        mock_response = {
            "status": "success",
            "jobs": [
                {
                    "job_id": "job_123",
                    "status": "DONE",
                    "backend": "ibm_brisbane",
                    "creation_date": "2024-01-01",
                },
                {
                    "job_id": "job_456",
                    "status": "RUNNING",
                    "backend": "ibm_kyoto",
                    "creation_date": "2024-01-02",
                },
            ],
            "total_jobs": 2,
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = list_my_jobs_sync(limit=10)

        assert result["status"] == "success"
        assert result["total_jobs"] == 2
        assert len(result["jobs"]) == 2

    def test_list_my_jobs_sync_failure(self, mocker):
        """Failed test jobs listing with sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to list jobs",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = list_my_jobs_sync(limit=10)

        assert result["status"] == "error"
        assert result["message"] == "Failed to list jobs"
        mocker_run_sync.assert_called_once()


class TestGetJobStatusSync:
    """Test get_job_status_sync function."""

    def test_get_job_status_sync_success(self, mocker):
        """Test successful job status retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "job_status": "DONE",
            "backend": "ibm_brisbane",
            "creation_date": "2024-01-01",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = get_job_status_sync("job_123")

        assert result["status"] == "success"
        assert result["job_id"] == "job_123"
        assert result["job_status"] == "DONE"

    def test_get_job_status_sync_failure(self, mocker):
        """Failed test job status retrieval with sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to get job status: service not initialized",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = get_job_status_sync("job_123")

        assert result["status"] == "error"
        assert result["message"] == "Failed to get job status: service not initialized"


class TestCancelJobSync:
    """Test cancel_job_sync function."""

    def test_cancel_job_sync_success(self, mocker):
        """Test successful job cancellation with sync wrapper."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "message": "Job cancellation requested",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = cancel_job_sync("job_123")

        assert result["status"] == "success"
        assert result["job_id"] == "job_123"
        mocker_run_sync.assert_called_once()

    def test_cancel_job_sync_failure(self, mocker):
        """Failed test job cancellation with sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to cancel job: service not initialized",
        }

        mocker_run_sync = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.sync._run_async"
        )
        mocker_run_sync.return_value = mock_response

        result = cancel_job_sync("job_123")

        assert result["status"] == "error"
        assert result["message"] == "Failed to cancel job: service not initialized"
        mocker_run_sync.assert_called_once()
