"""Given a number that is k digits long, extract the last two digits and the first two digits."""
n = int(input("Enter a number: "))
seqlen = int(input("Seqlen: "))
factor = 10 ** seqlen
nums = []
while n > 0:
    nums.append(n % factor)
    n //= factor

print(nums[::-1])
