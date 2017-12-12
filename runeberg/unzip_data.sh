#!/bin/bash
#wget https://files.slack.com/files-pri/T8AL5Q25U-F8B0A5LAX/download/data-20.zip?pub_secret=0135c07362 -O /tmp/data-20.zip
#wget https://files.slack.com/files-pri/T8AL5Q25U-F8B6PL2G5/download/data-19.zip?pub_secret=32a8da3526 -O /tmp/data-19.zip
#wget https://files.slack.com/files-pri/T8AL5Q25U-F8B3RU8DA/download/data-18.zip?pub_secret=41ef314a12 -O /tmp/data-18.zip

unzip /tmp/data-18.zip -d /tmp/data-18
unzip /tmp/data-19.zip -d /tmp/data-19
unzip /tmp/data-20.zip -d /tmp/data-20

rm /tmp/data-18.zip
rm /tmp/data-19.zip
rm /tmp/data-20.zip

mkdir /tmp/formated_data
mv /tmp/data-18/output/* /tmp/formated_data
mv /tmp/data-19/output/* /tmp/formated_data
mv /tmp/data-20/output/* /tmp/formated_data
