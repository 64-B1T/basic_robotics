from basic_robotics.interfaces import OPCUA_Client

def run_example():
    opc = OPCUA_Client('tcp://localhost:48010')
    try:
        opc.newComPort('Boiler1_FillLevel', "OPCEndpoint", ["0:Objects", "2:Demo","2:BoilerDemo","2:Boiler1", "2:FillLevelSensor", "2:FillLevel"])
        opc.newComPort('Boiler2_FillLevelSetpoint', "OPCEndpoint", ["0:Objects", "2:Demo","2:BoilerDemo","2:Boiler1", "2:FillLevelSetPoint"])

        opc.openCom()
        #print(opc.getData("Boiler1_temp"))
        print(opc.getData("Boiler1_FillLevel"))


        print(opc.getData("Boiler2_FillLevelSetpoint"))

        opc.sendData("Boiler2_FillLevelSetpoint", 7.9)
        print(opc.getData("Boiler2_FillLevelSetpoint"))
        opc.sendData("Boiler2_FillLevelSetpoint", 0.0)

        opc.closeCom()
    except Exception as e:
        print(e)
        opc.closeCom()