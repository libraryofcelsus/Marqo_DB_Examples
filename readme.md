# LLM Workspace
Version 0.01 of Marqo DB Examples by [LibraryofCelsus.com](https://www.libraryofcelsus.com)  
  
[Installation Guide](#installation-guide)  
[Skip to Changelog](#changelog)  
[Discord Server](https://discord.gg/pb5zcNa7zE)

------
**Recent Changes**

• 07/09 First Release, pretty basic for now.

------

### Current Scripts

All Scripts come with the ability to output to a dataset:

- **RAG Chatbot**: Simple RAG chatbot for talking with your own data.
- **File Processor**: Operates in the background to scan, chunk, clean, and upload files located in the ./Uploads directory to the database.
- **Long Term Memory Chatbot**: Still being worked on.
------


### What is this project?

Code examples from me experimenting with Marqo Vector DB.  

Aetherius Github: https://github.com/libraryofcelsus/Aetherius_AI_Assistant

------

My Ai work is self-funded by my day job, consider supporting me if you appreciate my work.

<a href='https://ko-fi.com/libraryofcelsus' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

------

Join the Discord for help or to get more in-depth information!

Discord Server: https://discord.gg/pb5zcNa7zE

Subscribe to my youtube for Video Tutorials: https://www.youtube.com/@LibraryofCelsus (Channel not Launched Yet)

Code Tutorials available at: https://www.libraryofcelsus.com/research/public/code-tutorials/

Made by: https://github.com/libraryofcelsus


------


## Future Plans

• Add Long Term Memory Chatbot

# Changelog: 
**0.01** 

• First Release


# Installation Guide

## Installer bat

Download the project zip folder by pressing the <> Code drop down menu.

**1.** Install Python 3.10.6, Make sure you add it to PATH: **https://www.python.org/downloads/release/python-3106/**

**2.** Run "install_requirements.bat" to install the needed dependencies.  

(If you get an error when installing requirements run: **python -m pip cache purge**)

**3.** Set up Marqo DB

Marqo Docs: https://docs.marqo.ai/2.9/  

To use a local Marqo server, first install Docker: https://www.docker.com.  
Next type: **docker pull marqoai/marqo:latest** in the command prompt.  
After it is finished downloading, type **docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest**   
(If it gives an error, check the docker containers tab for a new container and press the start button.  Sometimes it fails to start.)  

See: https://docs.docker.com/desktop/backup-and-restore/ for how to make backups.

Once the local Marqo DB server is running, it should be auto detected by the scripts.  

**6.** Install your desired API.  (Not needed if using OpenAi)  
https://github.com/LostRuins/koboldcpp  
https://github.com/oobabooga/text-generation-webui  

**8.** Launch a script with one of the **run_*.bat** 

**9.** Change the information inside of the "Settings" tab to your preferences.  This is where you can select what API to use, as well as set API keys.


**Additional Notes**  
To chat with your own files, use the File_Processor script.  It will generate a folder titled "Uploads".  Any file put here will be put into an external resource database to be used with the RAG Chatbot,

Photo OCR (jpg, jpeg, png) requires tesseract: https://github.com/UB-Mannheim/tesseract/wiki  
Once installed, copy the "Tesseract-OCR" folder from Program Files to the Main Project Folder.  




-----

## About Me

In January 2023, I had my inaugural experience with ChatGPT and LLMs in general. Since that moment, I've been deeply obsessed with AI, dedicating countless hours each day to studying it and to hands-on experimentation.

# Contact
Discord: libraryofcelsus      -> Old Username Style: Celsus#0262

MEGA Chat: https://mega.nz/C!pmNmEIZQ


