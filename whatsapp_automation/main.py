import pandas as pd
import tkinter as tk
from tkinter import filedialog
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
import urllib.parse

def get_excel_file():
    """Opens a file dialog to select an Excel file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select the Excel file with contacts",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    return file_path

def read_contacts(file_path):
    """Reads contacts from the specified Excel file."""
    if not file_path:
        print("No file selected. Exiting.")
        return None
    try:
        df = pd.read_excel(file_path)
        print("Successfully loaded contacts:")
        print(df)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return None

def initialize_driver():
    """Initializes the Chrome WebDriver using webdriver-manager."""
    try:
        print("Setting up ChromeDriver...")
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        driver.get("https://web.whatsapp.com")
        print("Please scan the QR code to log in to WhatsApp Web.")
        return driver
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        print("Please make sure you have Google Chrome installed.")
        return None

def send_whatsapp_message(driver, phone_number, message):
    """Sends a WhatsApp message to a given phone number."""
    try:
        # Wait for the user to log in by looking for the search bar
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-testid="chat-list-search"]')))

        # Format the phone number
        phone_str = str(phone_number)
        if phone_str.endswith('.0'):
            phone_str = phone_str[:-2]

        if len(phone_str) == 10 and phone_str.isdigit():
            phone_number = '+91' + phone_str
        elif not phone_str.startswith('+'):
            phone_number = '+' + phone_str
        else:
            phone_number = phone_str

        # Create the URL to open a chat
        url = f"https://web.whatsapp.com/send?phone={phone_number}"
        driver.get(url)

        # Wait for the message box to be ready and type the message
        message_box = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-testid="conversation-compose-box"]//div[@role="textbox"]'))
        )
        message_box.send_keys(message)

        # Click the send button
        send_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@data-testid="send"]'))
        )
        send_button.click()

        time.sleep(random.uniform(1, 3))  # Random delay
        return True, "Message sent successfully."
    except TimeoutException:
        error_message = f"Timed out waiting for element for {phone_number}. The number might be invalid or not on WhatsApp."
        print(error_message)
        return False, error_message
    except NoSuchElementException:
        error_message = f"Could not find an element for {phone_number}. The WhatsApp Web interface might have changed."
        print(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred for {phone_number}: {e}"
        print(error_message)
        return False, error_message

def main():
    """Main function to run the application."""
    excel_file = get_excel_file()
    contacts_df = read_contacts(excel_file)
    if contacts_df is not None:
        driver = initialize_driver()
        if driver:
            log = []
            for index, row in contacts_df.iterrows():
                first_name = row.get('First name', '')
                last_name = row.get('Last name', '')
                phone_number = row.get('Telephone number')
                message = row.get('Message', '')

                if pd.isna(phone_number) or not message:
                    print(f"Skipping row {index + 2}: Missing phone number or message.")
                    log.append({'phone_number': phone_number, 'status': 'Skipped - Missing data'})
                    continue

                # Personalize the message
                personalized_message = message.replace('{first_name}', str(first_name)).replace('{last_name}', str(last_name))

                print(f"Sending message to {first_name} {last_name} ({phone_number})...")
                success, status_message = send_whatsapp_message(driver, phone_number, personalized_message)

                if success:
                    print("Message sent successfully.")
                    log.append({'phone_number': phone_number, 'status': 'Success'})
                else:
                    print("Failed to send message.")
                    log.append({'phone_number': phone_number, 'status': status_message})

            driver.quit()
            log_df = pd.DataFrame(log)
            log_df.to_csv("message_log.csv", index=False)
            print("\nFinished sending messages. A log file 'message_log.csv' has been created.")

if __name__ == "__main__":
    main()
