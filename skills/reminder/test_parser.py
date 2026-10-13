import unittest
from datetime import datetime, timedelta

from skills.reminder.parser import parse_reminder_date, parse_time


class ReminderParserTests(unittest.TestCase):
  def test_tomorrow_afternoon(self) -> None:
    parsed = parse_reminder_date("tomorrow afternoon")
    self.assertIsNotNone(parsed)
    assert parsed is not None
    tomorrow = datetime.now().date() + timedelta(days=1)
    self.assertEqual(parsed.date(), tomorrow)
    self.assertEqual(parsed.hour, 15)

  def test_day_parts(self) -> None:
    for phrase, hour in (("tomorrow morning", 9), ("tomorrow evening", 18), ("tomorrow night", 21)):
      parsed = parse_reminder_date(phrase)
      self.assertIsNotNone(parsed)
      assert parsed is not None
      self.assertEqual(parsed.hour, hour)

  def test_explicit_time_remains_unchanged(self) -> None:
    parsed = parse_reminder_date("tomorrow at 3pm")
    self.assertIsNotNone(parsed)
    assert parsed is not None
    self.assertEqual(parsed.hour, 15)

  def test_explicit_duration_is_exact(self) -> None:
    self.assertEqual(parse_time("5 minutes"), 300)

  def test_compact_duration_is_exact(self) -> None:
    self.assertEqual(parse_time("10 secs"), 10)


if __name__ == "__main__":
  unittest.main()
