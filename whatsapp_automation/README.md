# WhatsApp Automation

This script automates sending personalized WhatsApp messages from an Excel file.

## Disclaimer

**Please be aware that automating WhatsApp messages may violate their Terms of Service and could result in your account being banned. Use this script at your own risk.**

## Setup

1.  **Install Python:** If you don't have Python installed, download and install it from [python.org](https://python.org).

2.  **Install Google Chrome:** This script uses the Chrome browser, so you'll need to have it installed.

3.  **Install Dependencies:**
    *   Open a terminal or command prompt.
    *   Navigate to the project directory (`whatsapp_automation`).
    *   Run the following command to install the required Python libraries:
        ```bash
        pip install -r requirements.txt
        ```
    *   The script will automatically download and manage the correct version of ChromeDriver for you.

## How to Run

1.  **Prepare your Excel file:**
    *   Create an Excel file with the following columns:
        *   `First name`
        *   `Last name`
        *   `Telephone number` (including the country code, e.g., +1234567890)
        *   `Message` (you can use `{first_name}` and `{last_name}` as placeholders for personalization)

2.  **Run the script:**
    *   Open a terminal or command prompt.
    *   Navigate to the project directory.
    *   Run the following command:
        ```bash
        python main.py
        ```

3.  **Log in to WhatsApp:**
    *   The script will open a new Chrome window with WhatsApp Web.
    *   Use your phone to scan the QR code and log in.

4.  **Select your Excel file:**
    *   A file dialog will open. Select the Excel file you prepared.

5.  **Let it run:**
    *   The script will start sending the messages one by one.
    *   Do not close the Chrome window until the script has finished.

6.  **Check the log:**
    *   Once the script is done, a `message_log.csv` file will be created in the project directory, showing the status of each message.
