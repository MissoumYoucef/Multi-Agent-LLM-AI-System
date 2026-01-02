"""
Alert Manager module.

Provides alerting for drift detection, quality degradation,
and system health issues.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """An alert instance."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 3600  # 1 hour default
    auto_resolve: bool = True


class AlertManager:
    """
    Manages alerts for system health and drift detection.

    Features:
    - Configurable alert rules
    - Alert cooldown to prevent spam
    - Webhook/notification support
    - Alert history and acknowledgment
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        max_history: int = 1000
    ):
        """
        Initialize alert manager.

        Args:
            webhook_url: Optional webhook URL for notifications.
            max_history: Maximum alerts to keep in history.
        """
        self.webhook_url = webhook_url
        self.max_history = max_history

        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=max_history)
        self._last_fired: Dict[str, float] = {}
        self._notification_handlers: List[Callable[[Alert], None]] = []

        self._alert_counter = 0

        logger.info("AlertManager initialized")

    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: str = "",
        cooldown_seconds: int = 3600
    ) -> None:
        """
        Add an alert rule.

        Args:
            name: Rule name.
            condition: Function that returns True when alert should fire.
            severity: Alert severity.
            message_template: Message template for the alert.
            cooldown_seconds: Minimum time between alerts.
        """
        self._rules[name] = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            message_template=message_template or f"Alert: {name}",
            cooldown_seconds=cooldown_seconds
        )
        logger.info(f"Added alert rule: {name}")

    def add_notification_handler(
        self,
        handler: Callable[[Alert], None]
    ) -> None:
        """Add a notification handler."""
        self._notification_handlers.append(handler)

    def check_rules(self) -> List[Alert]:
        """
        Check all rules and fire alerts if conditions met.

        Returns:
            List of newly fired alerts.
        """
        fired = []

        for name, rule in self._rules.items():
            # Check cooldown
            last = self._last_fired.get(name, 0)
            if time.time() - last < rule.cooldown_seconds:
                continue

            try:
                if rule.condition():
                    alert = self._fire_alert(rule)
                    fired.append(alert)
                elif rule.auto_resolve and name in self._active_alerts:
                    self.resolve_alert(self._active_alerts[name].id)
            except Exception as e:
                logger.error(f"Error checking rule {name}: {e}")

        return fired

    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: str = "",
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Manually fire an alert.

        Args:
            name: Alert name.
            severity: Alert severity.
            message: Alert message.
            source: Alert source.
            metadata: Additional metadata.

        Returns:
            The created Alert.
        """
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}"

        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message or f"Alert: {name}",
            source=source,
            metadata=metadata or {}
        )

        self._active_alerts[name] = alert
        self._alert_history.append(alert)
        self._last_fired[name] = time.time()

        self._notify(alert)

        logger.warning(f"Alert fired: {name} ({severity.value})")
        return alert

    def _fire_alert(self, rule: AlertRule) -> Alert:
        """Fire an alert from a rule."""
        return self.fire_alert(
            name=rule.name,
            severity=rule.severity,
            message=rule.message_template,
            source="rule"
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        note: str = ""
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID.
            note: Optional acknowledgment note.

        Returns:
            True if acknowledged.
        """
        for alert in self._active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                if note:
                    alert.metadata["ack_note"] = note
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve_alert(
        self,
        alert_id: str,
        resolution: str = ""
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID.
            resolution: Optional resolution note.

        Returns:
            True if resolved.
        """
        for name, alert in list(self._active_alerts.items()):
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                if resolution:
                    alert.metadata["resolution"] = resolution
                del self._active_alerts[name]
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        history = list(self._alert_history)

        if severity:
            history = [a for a in history if a.severity == severity]

        return history[-limit:]

    def _notify(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # Call registered handlers
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")

        # Send webhook if configured
        if self.webhook_url:
            self._send_webhook(alert)

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to webhook."""
        try:
            import httpx

            payload = {
                "alert_id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "source": alert.source
            }

            # Async send in background
            # In production, use proper async handling
            httpx.post(self.webhook_url, json=payload, timeout=5.0)

        except ImportError:
            logger.warning("httpx not installed for webhook")
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        by_severity = {}
        for alert in self._alert_history:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "active_alerts": len(self._active_alerts),
            "total_alerts": len(self._alert_history),
            "rules_count": len(self._rules),
            "by_severity": by_severity,
            "webhook_configured": self.webhook_url is not None
        }


# Pre-built alert rules for common scenarios
def create_drift_alert_rule(
    drift_detector,
    threshold: float = 0.3
) -> AlertRule:
    """Create a drift detection alert rule."""
    return AlertRule(
        name="semantic_drift",
        condition=lambda: drift_detector.get_drift_score() > threshold,
        severity=AlertSeverity.WARNING,
        message_template=f"Semantic drift exceeded threshold ({threshold})",
        cooldown_seconds=3600
    )


def create_quality_alert_rule(
    evaluator,
    threshold: float = 0.5
) -> AlertRule:
    """Create a quality degradation alert rule."""
    return AlertRule(
        name="quality_degradation",
        condition=lambda: not evaluator.check_quality_threshold(threshold),
        severity=AlertSeverity.CRITICAL,
        message_template=f"Response quality below threshold ({threshold})",
        cooldown_seconds=1800
    )


# Factory function
def create_alert_manager(
    webhook_url: Optional[str] = None
) -> AlertManager:
    """Create an alert manager."""
    return AlertManager(webhook_url=webhook_url)
