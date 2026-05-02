{{- define "rinlekha.namespace" -}}
{{ .Values.namespace }}
{{- end }}

{{- define "rinlekha.mlflowUrl" -}}
http://mlflow-service.{{ .Values.namespace }}.svc.cluster.local:{{ .Values.mlflow.port }}
{{- end }}

{{- define "rinlekha.vllmUrl" -}}
http://vllm-service.{{ .Values.namespace }}.svc.cluster.local:{{ .Values.vllm.port }}
{{- end }}
