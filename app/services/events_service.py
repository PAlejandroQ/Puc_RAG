"""
Events Service for RAG System.

Handles mock event generation and time series simulation for the system.
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List

from app.config import (
    AVAILABLE_TAGS,
    EVENT_TYPES,
    EVENT_SEVERITIES,
    EVENT_SEVERITY_WEIGHTS
)
from app.models import EventInfo

logger = logging.getLogger(__name__)


class EventsService:
    """Servicio para gestión de eventos simulados de series de tiempo"""

    def __init__(self):
        # Base de datos simulada de eventos por tipo con descripciones realistas
        self.event_templates = self._load_event_templates()

    def _load_event_templates(self) -> dict:
        """Cargar plantillas de eventos por tipo"""
        return {
            "Anomalía de Presión": [
                "Detectada anomalía de presión crítica. Valor {}% fuera del rango operativo normal.",
                "Pico de presión registrado por encima del límite de seguridad en sensor.",
                "Caída brusca de presión detectada. Posible pérdida de integridad del sistema.",
                "Oscilación de presión anormal registrada. Revisar estabilidad del flujo.",
                "Presión fuera de banda operacional por más de {} minutos consecutivos."
            ],
            "Pico de Temperatura": [
                "Temperatura elevada detectada. Sensor registra {}°C por encima del límite.",
                "Alerta térmica activada. Temperatura crítica en componente electrónico.",
                "Sobrecarga térmica detectada en sistema de control. Temperatura = {}°C.",
                "Aumento térmico anormal registrado. Verificar sistema de refrigeración.",
                "Temperatura del fluido por encima de parámetros operativos seguros."
            ],
            "Falla de Sensor": [
                "Falla de comunicación detectada en sensor. Pérdida de señal por {} minutos.",
                "Error de calibración automática detectado. Precisión comprometida.",
                "Ruido eléctrico anormal en señal del sensor. Filtrado aplicado.",
                "Desconexión temporal del sensor registrada. Sistema de respaldo activado.",
                "Falla intermitente detectada. Mantenimiento preventivo recomendado."
            ],
            "Cambio de Estado": [
                "Cambio de estado registrado en válvula ICV. Transición automática completada.",
                "Activación de protocolo de seguridad. Válvula DHSV cerrada preventivamente.",
                "Transición de modo operacional detectada. Sistema cambió a modo seguro.",
                "Cambio de configuración registrado. Parámetros operativos actualizados.",
                "Estado de válvula modificado por comando remoto."
            ],
            "Alerta de Seguridad": [
                "Condición de seguridad activada. Parámetros críticos excedidos.",
                "Alerta de sistema: condición de alto riesgo detectada.",
                "Protocolo de seguridad iniciado automáticamente.",
                "Falla de redundancia detectada. Sistema operativo con respaldo.",
                "Alerta crítica: intervención manual requerida inmediatamente."
            ],
            "Variación de Flujo": [
                "Variación significativa de flujo detectada. Cambio de {} m³/d.",
                "Anomalía en patrón de flujo registrada. Inestabilidad detectada.",
                "Cambio brusco en caudal medido. Revisar condiciones upstream.",
                "Flujo fuera de parámetros normales por período prolongado.",
                "Oscilación de producción detectada. Análisis de estabilidad requerido."
            ],
            "Mantenimiento Requerido": [
                "Alerta de desgaste: componente requiere mantenimiento preventivo.",
                "Ciclos de operación excedidos. Mantenimiento recomendado.",
                "Degradación de señal detectada. Calibración requerida.",
                "Acumulación de horas de operación. Inspección programada.",
                "Indicadores de condición sugieren mantenimiento inmediato."
            ],
            "Recuperación de Falla": [
                "Recuperación automática exitosa. Sistema restablecido normalmente.",
                "Falla corregida automáticamente. Funcionalidad restaurada.",
                "Sistema recuperado tras intervención automática.",
                "Normalización de parámetros completada exitosamente.",
                "Recuperación de comunicación establecida. Monitoreo continuado."
            ]
        }

    def generate_mock_events(self, start_date: str, end_date: str) -> List[EventInfo]:
        """
        Genera eventos simulados de series de tiempo para el período solicitado.

        Args:
            start_date: Fecha inicio en formato YYYY-MM-DD
            end_date: Fecha fin en formato YYYY-MM-DD

        Returns:
            Lista de eventos simulados
        """
        try:
            # Convertir fechas
            start = datetime.fromisoformat(start_date + "T00:00:00")
            end = datetime.fromisoformat(end_date + "T23:59:59")
        except ValueError as e:
            raise ValueError(f"Formato de fecha inválido: {e}")

        events = []
        event_id = 1000 + random.randint(1, 10000)

        # Generar entre 5-15 eventos para el período
        num_events = random.randint(5, 15)

        for i in range(num_events):
            # Timestamp aleatorio en el rango
            time_diff = (end - start).total_seconds()
            random_seconds = random.randint(0, int(time_diff))
            event_time = start + timedelta(seconds=random_seconds)

            # Seleccionar tag aleatorio
            tag_id = random.choice(AVAILABLE_TAGS)

            # Seleccionar tipo de evento aleatorio
            event_type = random.choice(EVENT_TYPES)

            # Seleccionar severidad basada en pesos
            severity = random.choices(EVENT_SEVERITIES, weights=EVENT_SEVERITY_WEIGHTS, k=1)[0]

            # Generar descripción específica
            template = random.choice(self.event_templates[event_type])

            # Rellenar placeholders en el template
            description = self._fill_template_placeholders(template)

            # Ajustar severidad basada en el contenido de la descripción
            if "crítica" in description.lower() or "alto riesgo" in description.lower():
                severity = "Critical"
            elif "seguridad" in description.lower() or "intervención" in description.lower():
                if severity == "Low":
                    severity = "Medium"

            event = EventInfo(
                event_id=event_id + i,
                tag_id=tag_id,
                timestamp=event_time.isoformat(),
                type=event_type,
                severity=severity,
                description=description
            )

            events.append(event)

        # Ordenar por timestamp
        events.sort(key=lambda x: x.timestamp)

        logger.info(f"Generados {len(events)} eventos simulados para el período {start_date} - {end_date}")
        return events

    def _fill_template_placeholders(self, template: str) -> str:
        """Rellenar placeholders en templates de eventos"""
        if "{}%" in template:
            value = random.randint(10, 200)
            return template.format(value)
        elif "°C" in template:
            value = random.randint(80, 150)
            return template.format(value)
        elif "m³/d" in template:
            value = random.randint(100, 2000)
            return template.format(value)
        elif "minutos" in template:
            value = random.randint(5, 120)
            return template.format(value)
        else:
            return template

    def get_available_tags(self) -> List[str]:
        """Retornar lista de tags disponibles para monitoreo"""
        return AVAILABLE_TAGS

    def get_event_types(self) -> List[str]:
        """Retornar tipos de eventos disponibles"""
        return EVENT_TYPES

    def get_event_severities(self) -> List[str]:
        """Retornar severidades disponibles"""
        return EVENT_SEVERITIES


# Instancia global del servicio
events_service = EventsService()