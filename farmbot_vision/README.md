# FarmBot Vision 0.2.1

Lightweight, experimental canopy measurement and safe FarmBot plant-radius recommendations. Open the app through Home Assistant Ingress after installation.

Set **Analysis resolution** (`640x480`, `960x720` default, or `1280x960`) in the app options; 960×720 is recommended for a 4 GB Raspberry Pi 4. Changing it requires an app restart.

The companion FarmBot integration must implement the service and event contract (`farmbot-vision-v2`) documented at repository level; the minimum compatible companion integration version is **1.2.0**. No FarmBot credential is accepted or stored by this app.

Upgrade from 0.2.0 by installing 0.2.1 and restarting the app. Close the old
browser tab and reopen the Web UI to create a fresh Home Assistant Ingress
session.
