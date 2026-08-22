"""Integracion con Claude (API directa de Anthropic) y verificacion.

Toda la dependencia de Anthropic se aisla en client.py. Modelos por rol
(router/sintesis/verificador) configurables por entorno. Modo 'simulado' para
probar el flujo sin clave ni gasto.
"""
