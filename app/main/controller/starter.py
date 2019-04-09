import speech_recognition as speech
import os
import sys
# import re
# import webbrowser
# import smtplib
# import requests
# import subprocess
# from pyowm import OWM
# import youtube_dl
# import vlc
# import urllib
# import urllib2
# import json
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
# import wikipedia
# import random
# from time import strftime
# import pyttsx3

from app.main.utils import logger


def start_desktop_voice_assistant():
    print('starter run')

    while True:
        assistant(retrieve_command())

    # for index, name in enumerate(speech.Microphone.list_microphone_names()):
    #     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))


def assistant(command):
    if 'hello' in command:
        convert_tts('Hello Serhiy')
    elif 'news for today' in command:
        try:
            news_url="https://news.google.com/news/rss"
            Client=urlopen(news_url)
            xml_page=Client.read()
            Client.close()
            soup_page=soup(xml_page, "xml")
            news_list=soup_page.findAll("item")
            for news in news_list[:5]:
                print(news.title.text)
                # print(news.title.text.encode('utf-8'))
                convert_tts(news.title.text)
        except Exception as e:
            print(e)
            convert_tts('Cannot tell you news - got some errors')
    elif 'stop' in command:
        convert_tts('Have a nice day')
        sys.exit()


def retrieve_command():
    recognizer = speech.Recognizer()

    with speech.Microphone() as source:
        print('Say something ...')
        recognizer.pause_threshold = 1
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        logger.info('User said: {}'.format(command))
    except speech.UnknownValueError:
        logger.error('UnknownValueError')
        command = retrieve_command()

    return command


def convert_tts(audio):
    for line in audio.splitlines():
        os.system("say " + line)

        # engine = pyttsx.init()
        # engine.say('Sally sells seashells by the seashore.')
        # engine.say('The quick brown fox jumped over the lazy dog.')
        # engine.runAndWait()



