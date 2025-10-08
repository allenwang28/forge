
import asyncio

from monarch.actor import Actor, endpoint, this_host

class Test(Actor):
    @endpoint
    def test(self):
        pass


async def main():
    p = this_host().spawn_procs(per_host={"d1": 2, "d2": 8})
    print(p.size())
    a = p.spawn("a", Test)
    print(a.size())


asyncio.run(main())