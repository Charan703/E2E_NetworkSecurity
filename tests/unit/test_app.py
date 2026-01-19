import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd
import io


class TestFastAPIApp:
    """Test FastAPI application endpoints"""
    
    @patch('app.client')  # Mock MongoDB client
    def test_health_endpoint(self, mock_client):
        """Test health check endpoint"""
        # Mock successful database ping
        mock_client.admin.command.return_value = True
        
        # Mock model files exist
        with patch('os.path.exists', return_value=True):
            from app import app
            client = TestClient(app)
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
    
    @patch('app.client')
    def test_stats_endpoint(self, mock_client):
        """Test statistics endpoint"""
        # Mock database collection count
        mock_collection = Mock()
        mock_collection.count_documents.return_value = 1000
        
        with patch('app.data_ingestion_collection', mock_collection):
            from app import app
            client = TestClient(app)
            
            response = client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_records" in data
            assert "model_accuracy" in data
            assert data["total_records"] == 1000
    
    def test_home_endpoint(self):
        """Test home page endpoint"""
        with patch('app.client'):  # Mock MongoDB client
            from app import app
            client = TestClient(app)
            
            response = client.get("/")
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
    
    @patch('app.load_object')
    @patch('app.client')
    def test_predict_endpoint_with_valid_csv(self, mock_client, mock_load):
        """Test prediction endpoint with valid CSV"""
        # Mock model and preprocessor
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_load.side_effect = [mock_preprocessor, mock_model]
        
        # Mock model prediction
        with patch('app.NetwrokModel') as mock_network_model:
            mock_instance = Mock()
            mock_instance.predict.return_value = [0, 1, 0]
            mock_network_model.return_value = mock_instance
            
            from app import app
            client = TestClient(app)
            
            # Create test CSV
            csv_content = "URL_Length,having_At_Symbol,SSLfinal_State\\n50,0,1\\n100,1,0\\n75,0,1"
            csv_file = io.StringIO(csv_content)
            
            response = client.post(
                "/predict",
                files={"file": ("test.csv", csv_file.getvalue(), "text/csv")}
            )
            
            assert response.status_code == 200
    
    def test_predict_endpoint_with_invalid_file(self):
        """Test prediction endpoint with invalid file type"""
        with patch('app.client'):  # Mock MongoDB client
            from app import app
            client = TestClient(app)
            
            response = client.post(
                "/predict",
                files={"file": ("test.txt", "invalid content", "text/plain")}
            )
            
            assert response.status_code == 400