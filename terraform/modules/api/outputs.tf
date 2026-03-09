output "api_endpoint" {
  value = aws_apigatewayv2_stage.main.invoke_url
}

output "alb_target_group_arn" {
  value = aws_lb_target_group.main.arn
}

output "alb_arn_suffix" {
  value = aws_lb.main.arn_suffix
}

output "waf_acl_arn" {
  value = aws_wafv2_web_acl.main.arn
}
