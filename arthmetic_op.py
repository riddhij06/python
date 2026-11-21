while True:

    a=int(input("enter a 1st number"))
    b=int (input("enter a 2nd number"))
    print("1.add 2.subtract 3.multiply 4.divide 5.exit")
    ch=int(input("enter choices"))
    if ch==1:
        print("1st num + 2nd num= ",a+b)
    elif ch==2:
        print("1st num - 2nd num= ",a-b)
    elif ch==3:
        print("1st num * 2nd num= ",a*b)
    elif ch==4:
        print("1st num / 2nd num= ",a/b)
    elif ch==5:
        print("exit")
    break