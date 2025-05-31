import ntplib
import time
from datetime import datetime
import socket
import platform
import os

class TimeSync:
    """Class to handle time synchronization and checking"""
    
    NTP_SERVERS = [
        'pool.ntp.org',
        'time.google.com',
        'time.windows.com',
        'time.apple.com'
    ]
    
    @staticmethod
    def get_ntp_time() -> datetime:
        """Get time from NTP server with fallback options"""
        client = ntplib.NTPClient()
        
        for server in TimeSync.NTP_SERVERS:
            try:
                response = client.request(server, timeout=5)
                return datetime.fromtimestamp(response.tx_time)
            except Exception as e:
                print(f"Failed to get time from {server}: {str(e)}")
                continue
        
        raise Exception("Failed to get NTP time from all servers")
    
    @staticmethod
    def get_system_time() -> datetime:
        """Get current system time"""
        return datetime.now()
    
    @staticmethod
    def check_time_drift() -> dict:
        """
        Check time drift between system and NTP time
        Returns dict with drift information
        """
        try:
            ntp_time = TimeSync.get_ntp_time()
            system_time = TimeSync.get_system_time()
            drift = abs((ntp_time - system_time).total_seconds())
            
            return {
                'system_time': system_time,
                'ntp_time': ntp_time,
                'drift_seconds': drift,
                'is_synchronized': drift < 60,  # Consider synchronized if drift < 1 minute
                'system_info': TimeSync.get_system_info()
            }
        except Exception as e:
            return {
                'error': str(e),
                'system_time': TimeSync.get_system_time(),
                'system_info': TimeSync.get_system_info()
            }
    
    @staticmethod
    def get_system_info() -> dict:
        """Get system information"""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'hostname': socket.gethostname(),
            'timezone': time.tzname
        }
    
    @staticmethod
    def sync_system_time():
        """
        Attempt to synchronize system time
        Note: Requires administrative privileges
        """
        try:
            if platform.system() == 'Windows':
                os.system('w32tm /resync')
                return "Windows time sync command executed"
            elif platform.system() == 'Linux':
                os.system('sudo ntpdate pool.ntp.org')
                return "Linux time sync command executed"
            elif platform.system() == 'Darwin':  # macOS
                os.system('sudo sntp -sS pool.ntp.org')
                return "macOS time sync command executed"
            else:
                return "Unsupported operating system"
        except Exception as e:
            return f"Failed to sync time: {str(e)}"

# Usage example
def main():
    """Main function to demonstrate time checking"""
    try:
        # Check time drift
        drift_info = TimeSync.check_time_drift()
        
        print("\n=== Time Synchronization Check ===")
        if 'error' in drift_info:
            print(f"Error: {drift_info['error']}")
            print(f"System Time: {drift_info['system_time']}")
        else:
            print(f"System Time: {drift_info['system_time']}")
            print(f"NTP Time: {drift_info['ntp_time']}")
            print(f"Time Drift: {drift_info['drift_seconds']:.2f} seconds")
            print(f"Is Synchronized: {drift_info['is_synchronized']}")
        
        print("\n=== System Information ===")
        system_info = drift_info['system_info']
        print(f"Platform: {system_info['platform']} {system_info['platform_release']}")
        print(f"Hostname: {system_info['hostname']}")
        print(f"Timezone: {system_info['timezone']}")
        
        # Attempt to sync if drift is too large
        if not drift_info.get('is_synchronized', False):
            print("\n=== Attempting Time Synchronization ===")
            sync_result = TimeSync.sync_system_time()
            print(sync_result)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
