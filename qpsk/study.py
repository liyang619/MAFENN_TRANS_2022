f=open('study.txt')
data=f.readlines()  #逐行读取txt并存成list。每行是list的一个元素，数据类型为str
l=[]
for i in range(len(data)):  #len(data)为数据行数
    l.append(float(data[i]))
print(l)
a = [9.955871062078958111e-01, 9.998617776241973676e-01, 9.999034443371603853e-01, 9.998969998855554708e-01, 9.998978887754319533e-01, 9.998967776630862669e-01, 9.999004443338270764e-01, 9.998987776653085469e-01, 9.999055554506172117e-01]
print(a)
print(type(a), type(l))