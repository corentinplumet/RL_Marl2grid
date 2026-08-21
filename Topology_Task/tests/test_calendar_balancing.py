import unittest

from teacher_student.calendar_balancing import (
    calendar_month_key,
    calendar_round_robin,
    format_month_counts,
)


class CalendarBalancingTest(unittest.TestCase):
    def test_extracts_month_from_datetime_and_scenario_name(self):
        self.assertEqual(calendar_month_key("2012-04-03 00:00:00"), "04")
        self.assertEqual(calendar_month_key("/data/Scenario_august_194"), "08")
        self.assertEqual(calendar_month_key("unknown", "Scenario_dec_007"), "12")
        self.assertEqual(calendar_month_key("not-a-calendar-value"), "unknown")

    def test_round_robin_interleaves_months_deterministically(self):
        values = [
            "Scenario_april_0",
            "Scenario_april_1",
            "Scenario_april_2",
            "Scenario_august_0",
            "Scenario_august_1",
            "Scenario_december_0",
            "Scenario_december_1",
        ]
        key = calendar_month_key
        first, counts = calendar_round_robin(values, key=key, seed=17)
        second, _ = calendar_round_robin(values, key=key, seed=17)

        self.assertEqual(first, second)
        self.assertCountEqual(first, values)
        self.assertEqual(counts, {"04": 3, "08": 2, "12": 2})
        self.assertEqual(
            {calendar_month_key(value) for value in first[:3]},
            {"04", "08", "12"},
        )

    def test_formats_month_counts(self):
        self.assertEqual(format_month_counts({"12": 3, "04": 7}), "04:7,12:3")


if __name__ == "__main__":
    unittest.main()
