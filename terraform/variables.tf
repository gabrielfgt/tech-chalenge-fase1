variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used as prefix for all resources"
  type        = string
  default     = "avc-predictor"
}

variable "google_api_key" {
  description = "Google Gemini API key"
  type        = string
  sensitive   = true
}

variable "api_image" {
  description = "Docker image URI for the FastAPI service (ECR)"
  type        = string
}

variable "ui_image" {
  description = "Docker image URI for the Streamlit service (ECR)"
  type        = string
}
