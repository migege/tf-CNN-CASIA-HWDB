#!/usr/bin/env python
# -*- coding:utf8 -*-
from flask import Flask

app = Flask(__name__)
from app import views
