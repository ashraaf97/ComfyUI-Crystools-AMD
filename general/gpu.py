import sys
import os
import torch
import comfy.model_management
from ..core import logger
import platform


def is_jetson() -> bool:
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception:
        platform_release = platform.release()
        logger.info(f"Platform release: {platform_release}")
        if 'tegra' in platform_release.lower():
            logger.info("Detected 'tegra' in platform release. Assuming Jetson device.")
            return True
        else:
            logger.info("JETSON: Not detected.")
            return False

IS_JETSON = is_jetson()


def _load_amdsmi():
    rocm_smi_path = '/opt/rocm/share/amd_smi'
    if rocm_smi_path not in sys.path and os.path.isdir(rocm_smi_path):
        sys.path.insert(0, rocm_smi_path)
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        return amdsmi
    except (ImportError, Exception):
        return None


class CGPUInfo:
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    amdsmiLoaded = False
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []

    def __init__(self):
        if IS_JETSON:
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('jtop initialized on Jetson device.')
            except ImportError as e:
                logger.error('jtop is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not initialize jtop. ' + str(e))
        else:
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                logger.info('pynvml (NVIDIA) initialized.')
            except ImportError:
                logger.info('pynvml not available, trying AMD SMI...')
            except Exception as e:
                logger.info(f'pynvml init failed ({e}), trying AMD SMI...')

            if not self.pynvmlLoaded:
                amdsmi = _load_amdsmi()
                if amdsmi is not None:
                    self.amdsmi = amdsmi
                    self.amdsmiLoaded = True
                    try:
                        self._amd_handles = amdsmi.amdsmi_get_processor_handles()
                        logger.info(f'amdsmi (AMD ROCm) initialized. Found {len(self._amd_handles)} GPU(s).')
                    except Exception as e:
                        logger.error('amdsmi: Could not get processor handles. ' + str(e))
                        self.amdsmiLoaded = False
                else:
                    logger.warning('Neither pynvml nor amdsmi could be loaded.')

        self.anygpuLoaded = self.pynvmlLoaded or self.jtopLoaded or self.amdsmiLoaded

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Could not pick default device. ' + str(e))

        if self.pynvmlLoaded and not self.jtopLoaded and not self.amdsmiLoaded and not self.deviceGetCount():
            logger.warning('No GPU detected, disabling GPU monitoring.')
            self.anygpuLoaded = False
            self.pynvmlLoaded = False

        if self.anygpuLoaded:
            if self.deviceGetCount() > 0:
                self.cudaDevicesFound = self.deviceGetCount()

                logger.info(f"GPU/s:")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"{deviceIndex}) {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                logger.warning('No GPU detected.')
        else:
            logger.warning('No GPU monitoring libraries available.')

        self.cudaDevice = 'cpu' if self.torchDevice == 'cpu' else 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

        if self.cuda and self.cudaAvailable and self.torchDevice == 'cpu':
            logger.warning('CUDA/ROCm is available, but torch is using CPU.')

    def getInfo(self):
        logger.debug('Getting GPUs info...')
        return self.gpus

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ''
        gpus = []

        if self.cudaDevice == 'cpu':
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        else:
            gpuType = self.cudaDevice

            if self.anygpuLoaded and self.cuda and self.cudaAvailable:
                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    gpuTemperature = -1

                    if self.switchGPU and self.gpusUtilization[deviceIndex]:
                        try:
                            gpuUtilization = self.deviceGetUtilizationRates(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU utilization. ' + str(e))
                            logger.error('Monitor of GPU is turning off.')
                            self.switchGPU = False

                    if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                        try:
                            memory = self.deviceGetMemoryInfo(deviceHandle)
                            vramUsed = memory['used']
                            vramTotal = memory['total']

                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.error('Could not get GPU memory info. ' + str(e))
                            self.switchVRAM = False

                    if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                        try:
                            gpuTemperature = self.deviceGetTemperature(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU temperature. Turning off this feature. ' + str(e))
                            self.switchTemperature = False

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                    })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def deviceGetCount(self):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetCount()
        elif self.amdsmiLoaded:
            return len(self._amd_handles)
        elif self.jtopLoaded:
            return 1
        else:
            return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        elif self.amdsmiLoaded:
            return self._amd_handles[index]
        elif self.jtopLoaded:
            return index
        else:
            return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.pynvmlLoaded:
            gpuName = 'Unknown GPU'
            try:
                gpuName = self.pynvml.nvmlDeviceGetName(deviceHandle)
                try:
                    gpuName = gpuName.decode('utf-8', errors='ignore')
                except AttributeError:
                    pass
            except UnicodeDecodeError as e:
                gpuName = 'Unknown GPU (decoding error)'
                logger.error(f"UnicodeDecodeError: {e}")
            return gpuName
        elif self.amdsmiLoaded:
            try:
                asic_info = self.amdsmi.amdsmi_get_gpu_asic_info(deviceHandle)
                return asic_info.get('market_name', 'AMD GPU')
            except Exception as e:
                logger.error('amdsmi: Could not get GPU name. ' + str(e))
                return 'AMD GPU'
        elif self.jtopLoaded:
            try:
                gpu_info = self.jtopInstance.gpu
                gpu_name = next(iter(gpu_info.keys()))
                return gpu_name
            except Exception as e:
                logger.error('Could not get GPU name. ' + str(e))
                return 'Unknown GPU'
        else:
            return ''

    def systemGetDriverVersion(self):
        if self.pynvmlLoaded:
            return f'NVIDIA Driver: {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.amdsmiLoaded:
            try:
                handle = self._amd_handles[0]
                driver_info = self.amdsmi.amdsmi_get_gpu_driver_info(handle)
                version = driver_info.get('driver_version', 'unknown')
                return f'AMD ROCm Driver: {version}'
            except Exception:
                return 'AMD ROCm Driver: unknown'
        elif self.jtopLoaded:
            return 'NVIDIA Driver: unknown'
        else:
            return 'Driver unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle).gpu
        elif self.amdsmiLoaded:
            try:
                activity = self.amdsmi.amdsmi_get_gpu_activity(deviceHandle)
                return activity.get('gfx_activity', -1)
            except Exception as e:
                logger.error('amdsmi: Could not get GPU utilization. ' + str(e))
                return -1
        elif self.jtopLoaded:
            try:
                gpu_util = self.jtopInstance.stats.get('GPU', -1)
                return gpu_util
            except Exception as e:
                logger.error('Could not get GPU utilization. ' + str(e))
                return -1
        else:
            return 0

    def deviceGetMemoryInfo(self, deviceHandle):
        if self.pynvmlLoaded:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
            return {'total': mem.total, 'used': mem.used}
        elif self.amdsmiLoaded:
            try:
                total = self.amdsmi.amdsmi_get_gpu_memory_total(
                    deviceHandle, self.amdsmi.AmdSmiMemoryType.VRAM
                )
                used = self.amdsmi.amdsmi_get_gpu_memory_usage(
                    deviceHandle, self.amdsmi.AmdSmiMemoryType.VRAM
                )
                return {'total': total, 'used': used}
            except Exception as e:
                logger.error('amdsmi: Could not get GPU memory info. ' + str(e))
                return {'total': 1, 'used': 1}
        elif self.jtopLoaded:
            mem_data = self.jtopInstance.memory['RAM']
            total = mem_data['tot']
            used = mem_data['used']
            return {'total': total, 'used': used}
        else:
            return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetTemperature(deviceHandle, self.pynvml.NVML_TEMPERATURE_GPU)
        elif self.amdsmiLoaded:
            try:
                temp = self.amdsmi.amdsmi_get_temp_metric(
                    deviceHandle,
                    self.amdsmi.AmdSmiTemperatureType.EDGE,
                    self.amdsmi.AmdSmiTemperatureMetric.CURRENT
                )
                return temp
            except Exception as e:
                logger.error('amdsmi: Could not get GPU temperature. ' + str(e))
                return -1
        elif self.jtopLoaded:
            try:
                temperature = self.jtopInstance.stats.get('Temp gpu', -1)
                return temperature
            except Exception as e:
                logger.error('Could not get GPU temperature. ' + str(e))
                return -1
        else:
            return 0

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()
        if self.amdsmiLoaded:
            try:
                self.amdsmi.amdsmi_shut_down()
            except Exception:
                pass
