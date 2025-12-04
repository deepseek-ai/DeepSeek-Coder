import unittest
from unittest.mock import MagicMock, patch, call
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import pandas as pd
import io
from selenium.webdriver.common.by import By
import pandas as pd

# This is a bit of a hack to make the import work without changing the project structure
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import main

class TestSendMessage(unittest.TestCase):

    @patch('main.WebDriverWait')
    def test_tc01_send_whatsapp_message_sends_keys_and_clicks_send(self, mock_wait):
    def test_send_whatsapp_message_sends_keys_and_clicks_send(self, mock_wait):
        # Arrange
        mock_driver = MagicMock()
        mock_message_box = MagicMock()
        mock_send_button = MagicMock()

        # Configure the mock_wait to return the mock elements for each of the 3 waits.
        # Configure the mock_wait to return the mock elements.
        # We use side_effect to return different mock objects for different waits.
        mock_wait.return_value.until.side_effect = [
            MagicMock(),      # For the initial chat-list-search wait
            mock_message_box, # For the conversation-compose-box wait
            mock_send_button  # For the send button wait
        ]

        phone_number = "1234567890"
        message = "Hello, this is a test message."

        # Act
        success, status_message = main.send_whatsapp_message(mock_driver, phone_number, message)

        # Assert
        mock_message_box.send_keys.assert_called_once_with(message)
        mock_send_button.click.assert_called_once()
        self.assertTrue(success)
        self.assertEqual(status_message, "Message sent successfully.")

    @patch('main.WebDriverWait')
    def test_tc02_send_whatsapp_message_handles_timeout(self, mock_wait):
        # Arrange
        mock_driver = MagicMock()

        # Simulate a TimeoutException when waiting for an element
        mock_wait.return_value.until.side_effect = [
            MagicMock(), # For the initial chat-list-search wait
            TimeoutException()
        ]

        phone_number = "1234567890"
        message = "This message will fail."

        # Act
        success, status_message = main.send_whatsapp_message(mock_driver, phone_number, message)

        # Assert
        self.assertFalse(success)
        self.assertIn("Timed out", status_message)
        self.assertIn(phone_number, status_message)

    @patch('main.webdriver.Chrome')
    @patch('main.WebDriverWait')
    def test_tc03_phone_number_formatting(self, mock_wait, mock_driver):
        # Arrange
        phone_number = "919876543210"
        expected_url = f"https://web.whatsapp.com/send?phone=+{phone_number}"

        # Act
        main.send_whatsapp_message(mock_driver, phone_number, "message")

        # Assert
        # Check that the driver was called with the correctly formatted number
        self.assertIn(call.get(expected_url), mock_driver.method_calls)

class TestDataHandling(unittest.TestCase):

    @patch('pandas.read_excel')
    def test_tc04_valid_excel_file_reading(self, mock_read_excel):
        # Arrange
        # Create a dummy dataframe that looks like the real data
        d = {'First name': ['test'], 'Last name': ['user'], 'Telephone number': [123], 'Message': ['hello']}
        df = pd.DataFrame(data=d)
        mock_read_excel.return_value = df

        # Act
        result = main.read_contacts("dummy_path.xlsx")

        # Assert
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, df)

    @patch('pandas.read_excel', side_effect=FileNotFoundError)
    def test_tc05_missing_excel_file(self, mock_read_excel):
        # Act
        result = main.read_contacts("non_existent_path.xlsx")

        # Assert
        self.assertIsNone(result)

        success = main.send_whatsapp_message(mock_driver, phone_number, message)

        # Assert
        # 1. Check if the correct URL was opened
        mock_driver.get.assert_called_once_with(f"https://web.whatsapp.com/send?phone=+{phone_number}")

        # 2. Check the waits for elements
        mock_driver_wait_calls = [
            call(mock_driver, 60),
            call(mock_driver, 30),
            call(mock_driver, 10),
        ]
        self.assertEqual(mock_wait.call_args_list, mock_driver_wait_calls)


        # 3. Check if send_keys was called on the message box
        mock_message_box.send_keys.assert_called_once_with(message)

        # 4. Check if the send button was clicked
        mock_send_button.click.assert_called_once()

        # 5. Check the function's return value
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
