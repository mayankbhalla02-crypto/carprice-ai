CAR PRICE AI
============

START IT (first run installs what it needs, so connect to the internet once)

  Windows   Double-click  CarPriceAI.bat
  Mac       Double-click  CarPriceAI.command
              (first time only: right-click the file, choose Open, then Open again)
  Linux     Run           ./CarPriceAI.sh

Your browser opens by itself. Sign in with the demo account, or create your own:
  Email:     demo@carai.com
  Password:  demo123

Needs Python 3.9 or newer (https://www.python.org/downloads/). On Windows, tick
"Add python.exe to PATH" in the installer.


USE IT FROM A PHONE OR TABLET
  Join the phone to the same Wi-Fi as the computer running the app, then open the
  "On your phone/tablet" address printed in the start-up window.


WHAT IS IN THIS FOLDER
  app.py                 the web app
  ml_model.py            loads car_model_rf.pkl and makes the predictions
  Cardekho.csv           the car data the model was trained on
  car_model_rf.pkl       your trained model
  carusers.db            accounts and estimate history
  run.py                 start-up helper used by the three launchers

NOTES
  * The model rates the cars listed in Cardekho.csv. A car that is not in that
    file is rejected with suggestions, never guessed.
  * If car_model_rf.pkl cannot be used on a computer (for example a different
    scikit-learn version), the app rebuilds the same model from Cardekho.csv and
    saves it as car_model_rf.local.pkl. Your original file is not changed.
  * To retrain with new data, replace Cardekho.csv (same columns). The model is
    rebuilt automatically on the next start.
  * To stop the app, close the window or press Ctrl+C.
