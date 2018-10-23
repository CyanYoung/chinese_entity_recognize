usr=`whoami`

if [ "$usr" == "cyan" ]
then
    pid=`ps -ef | grep Python | grep web_interface | awk '{print $2}'`
    if [ -n "$pid" ]
    then
        kill ${pid}
    fi
    nohup python3 web_interface.py >/dev/null 2>&1 &
else
    pid=`ps -ef | grep python | grep web_interface | awk '{print $2}'`
    if [ -n "$pid" ]
    then
        kill ${pid}
    fi
    nohup python web_interface.py -host 172.17.0.12 -port 2018 >/dev/null 2>&1 &
fi