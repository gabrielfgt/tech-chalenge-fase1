output "ui_url" {
  description = "URL da interface Streamlit"
  value       = "http://${aws_lb.main.dns_name}"
}

output "api_url" {
  description = "URL da API FastAPI"
  value       = "http://${aws_lb.main.dns_name}:8000"
}

output "api_docs_url" {
  description = "Swagger UI da API"
  value       = "http://${aws_lb.main.dns_name}:8000/docs"
}
