usr=`whoami`

if [[ "$usr" == "cyan" ]]
then
    pid=`ps -ef | grep Python | grep interface | awk '{print $2}'`
    if [[ -n "$pid" ]]
    then
        kill ${pid}
    fi
    nohup python3 interface.py >/dev/null 2>&1 &
else
    pid=`ps -ef | grep python | grep interface | awk '{print $2}'`
    if [[ -n "$pid" ]]
    then
        kill ${pid}
    fi
    nohup python interface.py -host 172.17.0.12 -port 2000 >/dev/null 2>&1 &
fi