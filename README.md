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
  Dear [Your Name],

  We regret to inform you that your account security has been breached. Our system detected unauthorized access from an unknown location. To secure your account, please click the link below to verify your identity:

  Verify Account Now

  Failure to take action within 24 hours will result in permanent account suspension. Don't delay; act now!

  Best regards,
  The Security Team
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
  subject: [CPSC 490.01] Important Notice: No Class and Final Proposal Due on upcoming Tuesday (May 7)

  CSUF external service. Use caution and confirm sender.

  Dear everyone,

  I would like to inform you that there will be no class on the upcoming Tuesday, May 7. Instead, your only remaining task for this course is to submit your group Final Proposal by midnight on May 7 via Canvas.

  Additionally, please note that there will be no final exam for this course. You can focus your efforts on preparing for your other classes during the final exam week.

  Thank you for your dedication throughout the semester. If you have any questions or need further clarification, please do not hesitate to reach out.


  Best regards,

  Rong Jin
  ```


* non spam
  ```
  Hey everyone,

  Homework 8 should now be posted. It mostly focuses on Machine Learning, although there are little bits from the section on neural networks so you may want to keep those notes handy. It's the standard quiz-style assignment with 60 minutes and 3 attempts. Please let me know if you have any questions, and I hope you all have a great weekend!

  Take care,

  Jeff
  ```


* non spam
  ```
  Hey everyone,

  I've posted a study guide for the final exam that we'll walk through in our review session tomorrow. Link

  I've also posted the assignment on canvas where you can upload your submission (report document and code .zip file) for your final projects. Bear in mind those are due THIS FRIDAY 5/10

  Hopefully that's all pretty straightforward, but please let me know if you have any questions. We'll talk more about this in class tomorrow, and I'm planning to do a more traditional review session walking through the topics one at a time. That said, if the class wants me to focus on a particular topic I'll be more than happy to.

  Take care, and I look forward to seeing you all tomorrow,

  Jeff
  ```