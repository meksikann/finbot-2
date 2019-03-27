## Finance-bot
##### Demo-bot to manage your personal income and spendings


###### Use python 3.5 virtual env
- install deps* ``pip install -r requirements.txt``
- download GLOVE from  https://nlp.stanford.edu/projects/glove/ unzip and paste glove.6B.100d.txt in ``app/main/ai/data/glove`` 
- training data path for NLU - ``app/main/ai/data/intents.json`` 
- train nlu model run ``python manage.py train_nlu``
- training data path for Dialog Management - ``app/main/ai/data/dialogs.yaml`` 
- train dialog model ``python manage.py train_dialog`` 
- paste slack credentials ``app/main/ai/creds/credentials.yaml``  it will remove ``dialog_state.pkl``
  ```
  slack:
    slack_token: "slack-token-goes-here"
    ```
- run server ``python manage.py run`` , port 8282
- to restart Dialog state tracker run: ``python manage.py restart_predictor``

- for local testing use ngrok ``./ngrok http 8282 -host-header="localhost:8282"  ``
- to use bot in slack - create slack app https://api.slack.com/slack-apps (don't forget include ngrko generated url in slack Event subscriptions and Interactive components) 

