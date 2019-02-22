## finance-bot
##### the demo-bot for managing your personal income and spendings


Use python 3.5 virtual env
0. Install deps* ``pip install requirements.txt``
- download GLOVE from  https://nlp.stanford.edu/projects/glove/ unzip and paste glove.6B.100d.txt in ``app/main/ai/data/glove`` 
- Install redis-server on Ubuntu ``sudo apt install redis-server``
- Start redis server ``sudo systemctl start redis``
- To run tests ``python manage.py test``
- Train nlu model ``python manage.py train_nlu``
- Train dialog model ``python manage.py train_dialog`` 
- Run server ``python manage.py run``

