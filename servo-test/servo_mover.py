import serial
import time
import sys

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200

# 명령 코드
CMD_MOVE_TIME_WRITE = 1
CMD_POS_READ = 28
CMD_LOAD_OR_UNLOAD_WRITE = 31
CMD_LOAD_OR_UNLOAD_READ = 32

class LX16AMover:
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
        except Exception as e:
            print(f"포트 열기 실패: {e}")
            sys.exit(1)

    def _send_cmd(self, s_id, cmd, params=[]):
        length = len(params) + 3
        packet = [0x55, 0x55, s_id, length, cmd] + params
        packet.append((~(sum(packet[2:]) & 0xff)) & 0xff)
        self.ser.write(bytearray(packet))

    def _recv(self):
        start = time.time()
        while (time.time() - start) < 0.1:
            if self.ser.in_waiting >= 6:
                header = self.ser.read(2)
                if header != b'\x55\x55': continue
                s_id, length, cmd = self.ser.read(3)
                data = list(self.ser.read(length - 3))
                self.ser.read(1) # Checksum
                return data
        return None

    def move(self, s_id, pos, duration=500):
        pos = max(0, min(1000, pos))
        params = [pos & 0xff, pos >> 8, duration & 0xff, duration >> 8]
        self._send_cmd(s_id, CMD_MOVE_TIME_WRITE, params)
        print(f">> {pos} 위치로 이동 ({duration}ms)")

    def set_torque(self, s_id, enable):
        val = 1 if enable else 0
        self._send_cmd(s_id, CMD_LOAD_OR_UNLOAD_WRITE, [val])
        print(f">> 토크 {'ON' if enable else 'OFF'}")

    def get_status(self, s_id):
        # 위치 읽기
        self._send_cmd(s_id, CMD_POS_READ)
        pos_data = self._recv()
        pos = "N/A"
        if pos_data:
            pos = pos_data[0] + (pos_data[1] << 8)
            if pos > 32767: pos -= 65536
        
        # 토크 상태 읽기
        self._send_cmd(s_id, CMD_LOAD_OR_UNLOAD_READ)
        torque_data = self._recv()
        torque = "N/A"
        if torque_data:
            torque = "ON" if torque_data[0] == 1 else "OFF"
            
        return pos, torque

    def close(self):
        self.ser.close()

def main():
    mover = LX16AMover(SERIAL_PORT, BAUD_RATE)
    
    try:
        s_id_input = input("제어할 서보 ID 입력: ").strip()
        if not s_id_input: return
        target_id = int(s_id_input)
        
        print(f"\n--- ID {target_id} 제어 모드 ---")
        print("명령: [위치](0-1000), on, off, r(상태읽기), q(종료)")

        while True:
            line = input(f"\n[ID {target_id}] 명령 >> ").strip().lower().split()
            if not line: continue
            cmd = line[0]
            
            if cmd == 'q': break
            elif cmd == 'on': mover.set_torque(target_id, True)
            elif cmd == 'off': mover.set_torque(target_id, False)
            elif cmd == 'r':
                pos, torque = mover.get_status(target_id)
                print(f">> 위치: {pos} | 토크: {torque}")
            else:
                try:
                    pos = int(cmd)
                    mover.move(target_id, pos)
                except ValueError:
                    print("올바른 명령이나 숫자를 입력하세요.")
            
    except (KeyboardInterrupt, EOFError):
        print("\n종료")
    finally:
        mover.close()

if __name__ == "__main__":
    main()
