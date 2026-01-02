"""
Unit tests for CostController module.

Tests budget management, usage tracking, and cost analytics.
"""
import pytest
import os
import time
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestBudgetPeriod:
    """Tests for BudgetPeriod enum."""

    def test_period_values(self):
        """Test period enum values."""
        from src.utils.cost_controller import BudgetPeriod

        assert BudgetPeriod.HOURLY.value == "hourly"
        assert BudgetPeriod.DAILY.value == "daily"
        assert BudgetPeriod.WEEKLY.value == "weekly"
        assert BudgetPeriod.MONTHLY.value == "monthly"


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_creation(self):
        """Test UsageRecord creation."""
        from src.utils.cost_controller import UsageRecord

        record = UsageRecord(
            model="gemini-pro",
            input_tokens=100,
            output_tokens=50,
            cost=0.001
        )
        assert record.model == "gemini-pro"
        assert record.input_tokens == 100
        assert record.success is True


class TestCostControllerInit:
    """Tests for CostController initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from src.utils.cost_controller import CostController

        controller = CostController()
        assert len(controller.budgets) == 1

    def test_custom_budget_initialization(self):
        """Test initialization with custom budget."""
        from src.utils.cost_controller import CostController

        controller = CostController(daily_budget=25.0, alert_threshold=0.9)
        status = controller.get_budget_status()
        assert status["limit"] == 25.0


class TestRecordUsage:
    """Tests for record_usage method."""

    def test_record_single_usage(self):
        """Test recording single usage."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        controller.record_usage(
            model="gemini-pro",
            input_tokens=100,
            output_tokens=50,
            cost=0.001
        )

        usage = controller.get_usage(BudgetPeriod.DAILY)
        assert usage == 0.001

    def test_record_multiple_usages(self):
        """Test recording multiple usages."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        controller.record_usage("gemini-pro", 100, 50, 0.001)
        controller.record_usage("gemini-pro", 200, 100, 0.002)

        usage = controller.get_usage(BudgetPeriod.DAILY)
        assert usage == 0.003


class TestCheckBudget:
    """Tests for check_budget method."""

    def test_within_budget(self):
        """Should return True when within budget."""
        from src.utils.cost_controller import CostController

        controller = CostController(daily_budget=10.0, enable_hard_limit=True)
        can_proceed, message = controller.check_budget(0.5)

        assert can_proceed is True
        assert "Within budget" in message

    def test_exceeds_budget(self):
        """Should return False when budget exceeded."""
        from src.utils.cost_controller import CostController

        controller = CostController(daily_budget=1.0, enable_hard_limit=True)
        controller.record_usage("gemini-pro", 1000, 1000, 0.9)

        can_proceed, message = controller.check_budget(0.2)
        assert can_proceed is False
        assert "exceeded" in message.lower()

    def test_soft_limit_allows_proceed(self):
        """Soft limit should allow proceeding even when exceeded."""
        from src.utils.cost_controller import CostController

        controller = CostController(daily_budget=1.0, enable_hard_limit=False)
        controller.record_usage("gemini-pro", 1000, 1000, 1.5)

        can_proceed, _ = controller.check_budget(0.1)
        assert can_proceed is True


class TestGetUsageByModel:
    """Tests for get_usage_by_model method."""

    def test_usage_by_model(self):
        """Should return usage breakdown by model."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        controller.record_usage("gemini-pro", 100, 50, 0.001)
        controller.record_usage("gpt-4", 100, 50, 0.01)
        controller.record_usage("gemini-pro", 100, 50, 0.002)

        usage = controller.get_usage_by_model(BudgetPeriod.DAILY)
        assert usage["gemini-pro"] == 0.003
        assert usage["gpt-4"] == 0.01


class TestGetBudgetStatus:
    """Tests for get_budget_status method."""

    def test_status_structure(self):
        """Should return expected status structure."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController(daily_budget=10.0)
        status = controller.get_budget_status(BudgetPeriod.DAILY)

        assert "period" in status
        assert "limit" in status
        assert "used" in status
        assert "remaining" in status
        assert "percentage" in status

    def test_unconfigured_period(self):
        """Should return error for unconfigured period."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        status = controller.get_budget_status(BudgetPeriod.WEEKLY)

        assert "error" in status


class TestGetAnalytics:
    """Tests for get_analytics method."""

    def test_empty_analytics(self):
        """Should return empty analytics when no usage."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        analytics = controller.get_analytics(BudgetPeriod.DAILY)

        assert analytics["total_requests"] == 0
        assert analytics["total_cost"] == 0

    def test_analytics_with_usage(self):
        """Should return correct analytics with usage."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        controller.record_usage("gemini-pro", 100, 50, 0.001)
        controller.record_usage("gemini-pro", 200, 100, 0.002)

        analytics = controller.get_analytics(BudgetPeriod.DAILY)

        assert analytics["total_requests"] == 2
        assert analytics["total_cost"] == 0.003
        assert "by_model" in analytics


class TestAddBudget:
    """Tests for add_budget method."""

    def test_add_weekly_budget(self):
        """Should add weekly budget."""
        from src.utils.cost_controller import CostController, BudgetPeriod

        controller = CostController()
        controller.add_budget(BudgetPeriod.WEEKLY, 50.0)

        assert BudgetPeriod.WEEKLY in controller.budgets
        status = controller.get_budget_status(BudgetPeriod.WEEKLY)
        assert status["limit"] == 50.0


class TestCleanupOldRecords:
    """Tests for cleanup_old_records method."""

    def test_cleanup_removes_old(self):
        """Should remove old records."""
        from src.utils.cost_controller import CostController

        controller = CostController()
        # Add a record
        controller.record_usage("gemini-pro", 100, 50, 0.001)

        # Manually age the record
        controller._usage_log[0].timestamp = time.time() - (31 * 86400)

        removed = controller.cleanup_old_records(days=30)
        assert removed == 1


class TestCreateCostController:
    """Tests for create_cost_controller factory function."""

    def test_factory_function(self):
        """Should create controller with specified settings."""
        from src.utils.cost_controller import create_cost_controller

        controller = create_cost_controller(
            daily_budget=50.0,
            alert_threshold=0.75,
            enable_hard_limit=True
        )

        status = controller.get_budget_status()
        assert status["limit"] == 50.0
