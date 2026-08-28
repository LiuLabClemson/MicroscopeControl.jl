

using Revise
using MicroscopeControl
using MicroscopeControl.HardwareImplementations.XEM_DAC




fpga = XEM_dac()
#fpga.xem.bitfile = raw"C:\Users\sheng\Documents\Vivado\dac_slowclk_eod\dac_slowclk_eod.runs\dac_eod_dac1280ns.bit"
#fpga.xem.bitfile = raw"C:\Users\sheng\Documents\Vivado\dac_slowclk_eod\dac_slowclk_eod.runs\dac_eod_dac80ns.bit"
fpga.xem.bitfile = raw"C:\Users\sheng\Documents\Vivado\dac_slowclk_eod\dac_slowclk_eod.runs\impl_1\dac_eod.bit"
initialize(fpga.xem)

va = 1.0 # voltage for channel A
vb = -1.0 # voltage for channel B
vc = 1.0 # voltage for channel C
vd = -1.0 # voltage for channel D



va = -0.0
vb = -0.5
vc = -1.0
vd = -1.5

#code = XEM_DAC.volts_to_code(0.0)

setvoltageD(fpga, 0.5)


setvoltageAll(fpga, va,vb,vc,vd)

start(fpga)




stop(fpga)


# generate sawtooth wave on channel A and B





voltages = collect(-0.005:0.002:0.005)
cyclenum = 20
start(fpga)
for cycle in 1:cyclenum
    for v in voltages
        #setvoltageA(fpga, v)
        setvoltageB(fpga, v)
        #setvoltage(daq,t, v)
        sleep(0.001)
    end
end
stop(fpga)




shutdown(fpga.xem)












