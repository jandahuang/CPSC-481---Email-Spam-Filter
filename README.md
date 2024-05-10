# CPSC-481---Email-Spam-Filter

## Install
1. Open up your terminal and cd into the `interface` folder
2. Create a new virtual environment by running (command may differ slightly depending on python configuration):
   Windows:
   ```
   py -m venv .venv
   ```
   macOS/Linux:
   ```
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   Windows:
   ```
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   . .venv/bin/activate
   ```
4. Make sure pip is up to date:
   Windows:
   ```
   pip install --upgrade pip
   pip --version
   ```
   macOS/Linux:
   ```
   pip install --upgrade pip
   pip --version
   ```
5. Install from the `requirements.txt` file:
   Windows:
   ```
   pip install -r requirements.txt
   ```
   macOS/Linux:
   ```
   pip install -r requirements.txt
   ```
6. Train the data:
   ```
   py training.py
   ```
   macOS/Linux:
   ```
   python3 training.py
   ```
7. Run the server (see the "Run" section)

## Run
1. Initiated the Virual Environment (if not already initiated)
   Windows:
   ```
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   . .venv/bin/activate
   ```

2. Run Flask (with the virtual environment running)
   ```
   flask --app app run --debug
   ```
   macOS/Linux:
   ```
   flask --app app run --debug
   ```

3. Flask will now start a development server. Once running, it will tell you the URL and port it is running on. It will most likely be [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Go to the URL in the previous step

5. Enter the email into the text box (some sample emails are provided in the "Test Emails" section) and click "Submit"

6. The results will be displayed below the submit button along with the result of each algorithm

7. To shut down the server, in the terminal Flask is running in, enter CTRL + C (COMMAND + C for macOS)



## Test Emails


* Spam
  ```
  Dear [Your Name],

  Congratulations! You've been selected as the lucky winner of our Grand Prize Giveaway!

  You've won a brand-new iPhone 13, a luxury vacation to an exotic island, and $10,000 in cash!

  To claim your prize, simply click the link below and provide your personal details:

  Click here to claim your prize!

  Hurry, this offer expires in 24 hours! 

  Best regards,
  The Prize Patrol Team
  ```



* Spam
  ```
  Subject: "Forward This to 10 Friends or Bad Luck Will Follow You!"
  Dear [Your Name],

  You've received this email because you're one of the lucky few. If you break the chain, misfortune will haunt you forever. Forward this to 10 friends within the next hour, and good luck will come your way!

  Sincerely,
  Superstitious Sender
  ```

* Spam
  ```
  Subject: "Investment Opportunity: Guaranteed 1000% Returns!"
  Hello [Your Name],

  I'm a wealthy individual from a foreign country, and I need your help. I have $10 million locked up in a bank, and I need someone trustworthy to help me transfer it. If you provide your bank details and a small processing fee, I'll share a portion of the funds with you. Act fast; this opportunity won't last!

  Best regards,
  Prince Scamalot
  ```

  
  
* Non Spam
  ```
  Dear [Your Name],

  Thank you for your recent purchase! We're excited to let you know that your order #12345 has been successfully processed. Here are the details:

  - Product: XYZ Widget
  - Quantity: 2
  - Total Amount: $99.99

  If you have any questions or need further assistance, feel free to reply to this email. We appreciate your business!

  Best regards,
  The XYZ Store Team
  ```


* non spam
  ```
  Hello [Your Name],

  Thank you for subscribing to our newsletter! 🎉 You're now part of our community, and we'll keep you updated on the latest news, tips, and special offers.

  Stay tuned for exciting content delivered straight to your inbox. If you have any preferences or topics you'd like us to cover, feel free to reply to this email.

  Cheers,
  The Newsletter Team
  ```


* non spam
  ```
  Hi [Your Name],

  We received a request to reset your account password. If you initiated this, please click the link below to set a new password:

  Reset Password

  If you didn't request this, please ignore this email. Your account remains secure.

  Regards,
  The Support Team
  ```