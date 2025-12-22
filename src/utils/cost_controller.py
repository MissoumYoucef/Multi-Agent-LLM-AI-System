"""
Cost Controller module.

Provides budget limits, cost tracking, and usage monitoring
for LLM API calls.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class UsageRecord:
    """Record of a single API usage."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    session_id: str = "default"
    success: bool = True


@dataclass
class BudgetConfig:
    """Budget configuration."""
    limit: float  # USD
    period: BudgetPeriod
    alert_threshold: float = 0.8  # Alert at 80% of limit
    hard_limit: bool = True  # Whether to block when limit reached


class CostController:
    """
    Controls and monitors LLM API costs.
    
    Tracks usage, enforces budgets, and provides cost analytics.
    """
    
    def __init__(
        self,
        daily_budget: float = 10.0,
        alert_threshold: float = 0.8,
        enable_hard_limit: bool = False
    ):
        """
        Initialize cost controller.
        
        Args:
            daily_budget: Daily budget in USD.
            alert_threshold: Alert when usage reaches this fraction of budget.
            enable_hard_limit: Block requests when budget exceeded.
        """
        self.budgets: Dict[BudgetPeriod, BudgetConfig] = {
            BudgetPeriod.DAILY: BudgetConfig(
                limit=daily_budget,
                period=BudgetPeriod.DAILY,
                alert_threshold=alert_threshold,
                hard_limit=enable_hard_limit
            )
        }
        
        self._usage_log: List[UsageRecord] = []
        self._alerts_sent: Dict[str, float] = {}  # Alert key -> timestamp
        self._alert_cooldown = 3600  # 1 hour between same alerts
        
        logger.info(f"CostController initialized: daily_budget=${daily_budget}")
    
    def add_budget(
        self,
        period: BudgetPeriod,
        limit: float,
        alert_threshold: float = 0.8,
        hard_limit: bool = False
    ) -> None:
        """Add or update a budget."""
        self.budgets[period] = BudgetConfig(
            limit=limit,
            period=period,
            alert_threshold=alert_threshold,
            hard_limit=hard_limit
        )
    
    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        session_id: str = "default",
        success: bool = True
    ) -> None:
        """
        Record an API usage.
        
        Args:
            model: Model name used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost: Cost in USD.
            session_id: Session identifier.
            success: Whether the call was successful.
        """
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=session_id,
            success=success
        )
        self._usage_log.append(record)
        
        # Check budgets and trigger alerts
        self._check_budgets()
        
        logger.debug(f"Recorded usage: {model}, ${cost:.4f}")
    
    def check_budget(self, estimated_cost: float = 0) -> tuple[bool, str]:
        """
        Check if a request can proceed within budget.
        
        Args:
            estimated_cost: Estimated cost of upcoming request.
            
        Returns:
            Tuple of (can_proceed, message).
        """
        for period, config in self.budgets.items():
            if config.hard_limit:
                current = self.get_usage(period)
                if current + estimated_cost > config.limit:
                    msg = f"Budget exceeded for {period.value}: ${current:.2f}/${config.limit:.2f}"
                    logger.warning(msg)
                    return False, msg
        
        return True, "Within budget"
    
    def get_usage(self, period: BudgetPeriod = BudgetPeriod.DAILY) -> float:
        """
        Get total usage for a period.
        
        Args:
            period: The budget period.
            
        Returns:
            Total cost in USD.
        """
        cutoff = self._get_period_cutoff(period)
        return sum(
            r.cost for r in self._usage_log
            if r.timestamp >= cutoff
        )
    
    def get_usage_by_model(
        self,
        period: BudgetPeriod = BudgetPeriod.DAILY
    ) -> Dict[str, float]:
        """Get usage breakdown by model."""
        cutoff = self._get_period_cutoff(period)
        usage: Dict[str, float] = {}
        
        for record in self._usage_log:
            if record.timestamp >= cutoff:
                usage[record.model] = usage.get(record.model, 0) + record.cost
        
        return usage
    
    def get_budget_status(
        self,
        period: BudgetPeriod = BudgetPeriod.DAILY
    ) -> Dict[str, Any]:
        """Get budget status for a period."""
        config = self.budgets.get(period)
        if not config:
            return {"error": f"No budget configured for {period.value}"}
        
        current = self.get_usage(period)
        remaining = max(0, config.limit - current)
        percentage = (current / config.limit) * 100 if config.limit > 0 else 0
        
        return {
            "period": period.value,
            "limit": config.limit,
            "used": current,
            "remaining": remaining,
            "percentage": percentage,
            "alert_threshold_reached": percentage >= config.alert_threshold * 100,
            "limit_reached": current >= config.limit
        }
    
    def get_analytics(
        self,
        period: BudgetPeriod = BudgetPeriod.DAILY
    ) -> Dict[str, Any]:
        """Get detailed analytics for a period."""
        cutoff = self._get_period_cutoff(period)
        records = [r for r in self._usage_log if r.timestamp >= cutoff]
        
        if not records:
            return {
                "period": period.value,
                "total_requests": 0,
                "total_cost": 0,
                "avg_cost_per_request": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
                "by_session": {}
            }
        
        total_cost = sum(r.cost for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        
        by_model: Dict[str, Any] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {"requests": 0, "cost": 0, "tokens": 0}
            by_model[r.model]["requests"] += 1
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["tokens"] += r.input_tokens + r.output_tokens
        
        by_session: Dict[str, float] = {}
        for r in records:
            by_session[r.session_id] = by_session.get(r.session_id, 0) + r.cost
        
        return {
            "period": period.value,
            "total_requests": len(records),
            "successful_requests": sum(1 for r in records if r.success),
            "total_cost": total_cost,
            "avg_cost_per_request": total_cost / len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_model": by_model,
            "by_session": by_session
        }
    
    def _get_period_cutoff(self, period: BudgetPeriod) -> float:
        """Get timestamp cutoff for a period."""
        now = datetime.now()
        
        if period == BudgetPeriod.HOURLY:
            cutoff = now.replace(minute=0, second=0, microsecond=0)
        elif period == BudgetPeriod.DAILY:
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == BudgetPeriod.WEEKLY:
            cutoff = now - timedelta(days=now.weekday())
            cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # MONTHLY
            cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return cutoff.timestamp()
    
    def _check_budgets(self) -> None:
        """Check budgets and trigger alerts if needed."""
        for period, config in self.budgets.items():
            current = self.get_usage(period)
            percentage = current / config.limit if config.limit > 0 else 0
            
            if percentage >= config.alert_threshold:
                alert_key = f"{period.value}_threshold"
                self._send_alert(
                    alert_key,
                    f"Budget alert: {period.value} usage at {percentage*100:.1f}% "
                    f"(${current:.2f}/${config.limit:.2f})"
                )
    
    def _send_alert(self, key: str, message: str) -> None:
        """Send an alert if not in cooldown."""
        now = time.time()
        last_sent = self._alerts_sent.get(key, 0)
        
        if now - last_sent >= self._alert_cooldown:
            self._alerts_sent[key] = now
            logger.warning(f"COST ALERT: {message}")
            # In production, this would send to monitoring system
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Remove records older than specified days."""
        cutoff = time.time() - (days * 86400)
        original_count = len(self._usage_log)
        self._usage_log = [r for r in self._usage_log if r.timestamp >= cutoff]
        removed = original_count - len(self._usage_log)
        logger.info(f"Cleaned up {removed} old usage records")
        return removed


# Factory function
def create_cost_controller(
    daily_budget: float = 10.0,
    alert_threshold: float = 0.8,
    enable_hard_limit: bool = False
) -> CostController:
    """Create a cost controller with specified budget."""
    return CostController(
        daily_budget=daily_budget,
        alert_threshold=alert_threshold,
        enable_hard_limit=enable_hard_limit
    )
