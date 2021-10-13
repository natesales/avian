from networktables import NetworkTables


class SmartDashboard:
    def __init__(self, server, defaults, change_listener, change_key):
        self.defaults = defaults

        NetworkTables.initialize(server)
        self._sd = NetworkTables.getTable("SmartDashboard")

        for key in self.defaults:
            self._sd.putValue(key, self.defaults[key])

        NetworkTables.addConnectionListener(lambda connected, info: print(info, ",", connected), immediateNotify=True)
        self._sd.addEntryListener(change_listener, immediateNotify=True, key=change_key)

    def get(self, key: str):
        return self._sd.getValue(key, self.defaults[key])

    def set(self, key: str, val):
        return self._sd.putValue(key, val)
