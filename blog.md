
# The Herlihy Hierarchy

It's impossible to implement many concurrent algorithms without explicit hardware support! That is, the processor has to provide sufficiently rich instructions in order for you to construct many essential concurrent objects! How can we understand this impossibility result? In order to talk about it, we have to get some terminology out of the way...

## Setup, Linearizability
The idea is that we have several *threads* that all have access to one or more objects at the same time. Each object has some methods that you can call, and at some point that method will return with a response. For instance, a *register* has *get* and *set* methods that follow the expected sort of rules for recording data in memory. For instance, this is a valid sequence of calls for a register $r$:

1. r.set(3)
2. r.set(5)
3. r.get() // returns 5
4. r.get() // returns 5
5. r.set(10) ✅

And this is invalid...
1. r.set(10)
2. r.get() // returns 2 ❌

Things get interesting when multiple threads run these simultaneously. For instance, imagine a system where the values of registers are 8 bit arrays, and threads write one bit at a time. Thread A tries to write 00110111 and thread B tries to write 11011011. Initially, the value is 00000000. Look what might happen...

1. thread A starts and writes the first 4 bits, leaving 00000111. It then takes a little nap.
2. thread B starts and writes everything, leaving 11011011!
3. thread A wakes up and finishes its last four bits, leaving 00111011.


Now the resulting register holds something that is neither what is started with, nor what either thread wanted to write! Hence, objects that are meant to be shared between threads like this must have conditions on behavior to ensure sensible results. A strong one is called *linearlizability*. For a linearizable object, I must be able to rearrange all the calls and responses, subject to the constraints:

1. The resulting sequence of calls and responses is valid for the object
2. For each thread, the sequence of responses "seems correct" from its standpoint.
3. If one invocation finishes before another invocation starts, that invocation comes before the other.
4. Each invocation is followed by the appropriate response

A register satisfying this property is called *atomic*. A register implemented like above must NOT be atomic, because I must be able to rewrite the order of calls and responses to either: A starts write, A ends write, B starts write, B ends right or vice versa. In either case, the end result must be ONE of the two values A, B were trying to write!

## An atomic stack
Now, let's give one other example of a linearizable object. In this case, we want to implement a linearizable stack. The stack has methods `push` and `pop`, and after linearizing it according to the rules above, we should get the correct behavior of a single-threaded stack. We can't use a regular old stack for this purpose - I encourage you to try it out in your favorite language! Something simple like this:

```
void worker(int n_cycles, vector<int>* vec) {
	for (int i = 0; i < n_cycles; i++) {
		vec->push_back(i);
	}

	for (int i = 0; i < n_cycles; i++) {
		vec->pop_back();
	}
};
```

For a few thousand cycles and workers should break in all kinds of ways - workers might try to pop from an empty stack, not pop everything, all kinds of craziness! (Kind of like the 8 bit register from above...) So in order to get a stack the behaves sensibly, one way is to use a lock. For instance you do this:

```
class AtomicIntStack {
	mutex my_lock;
	vector<int> stack;
	public:
		void push(int x) {
			my_lock.lock();
			stack.push_back(x);
			my_lock.unlock();
		}

		int pop() {
			my_lock.lock();
			int out = stack.back();
			stack.pop_back();
			my_lock.unlock();
			return out
		}
}
```

The difference here is that we've wrapped the calls in a lock/unlock pair. When a thread successfully calls `lock` and returns, we say it *holds the lock* until it successfully calls `unlock` and returns. The point is that only one thread can hold the lock at once. All other threads that try to call `lock` just have to sit at that line, stuck, until it frees up. (Note that this is *not* how you should use locks in the real world... what would happen if the `pop_back` call raises an exception?)

With this guarantee that only one thread can hold the lock at a time, you can prove that the `AtomicIntStack` does satisfy the linearizability conditions, and we have built it from an ordinary stack and a lock.

## Wait Freedom
There is one more definition we need before getting to the main part. It's the notion of *wait freedom*. In the atomic stack described above, note that if a thread acquires the lock and then is killed without releasing it, no other thread can may any progress! To avoid this outcome we can demand our methods are *wait-free*, meaning that after finitely many steps of work by the calling thread, the method will return. The other threads can go to sleep, be killed, or also do work, but none of that matters. As long as a thread doesn't give up, it'll succeed!


## Consensus
Now, let's introduce the main concurrent object needed: consensus. A consensus object has one method: `propose(x)` that takes one argument `x`. It behaves in this way:

1. `propose` returns the same value for all threads that call it
2. `propose` returns the input value for some call

We're interested in seeing if we can construct *wait-free, linearizable* consensus objects that will work for N threads. The N=1 case is kind of boring, but things already get interesting for N=2:

```
A wait-free, linearizable consensus object for N=2 cannot be constructed using only atomic registers.
```

Pause and savor this statement for a moment. It's an impossibility result! If your CPU only provides you with registers with `get` and `set` methods that are atomic, it is *impossible to construct any nontrivial consensus object*! No matter how hard you try!

So why is this true? Let's simplify a bit and assume from now on the consensus protocol is binary - the argument for the `propose` method is 0 or 1. The values in the registers and any other program state, including the inputs, comprise some state `S`. From that state, either thread may "move" and take some action, changing the state, giving rise to a tree of states and actions. If all paths out of a state end up leading to the same final consensus value, we call it univalent. We call it 0 valent or 1 valent if the outcome is 0 or 1, respectively. If both are outcomes are still possible from a state, it is bivalent. We call a bivalent state *critical* if all paths out of it lead to univalent states. 

```
There is always some critical bivalent state.
```

So there has to be *some* bivalent initial state. To see this, suppose thread A proposes 0 and thread B proposes 1. Since the protocol is wait free, I must be allowed to run A in isolation, which means the outcome of `propose` must be a 0! Similarly, I must be allowed to run B in isolation until it terminates, which gives a path with the outcome 1. Now that we know there is some bivalent state, why must there be a critical one? Well, suppose that `S0` is bivalent and not critical. Then there is some action by a thread that leads to bivalent `S1`. If *that's* not critical, some action leads to `S2`, and so on. So if there is no bivalent critical state, you build up a chain `S0 -> S1 -> S2 -> S3 -> ...` of infinitely many bivalent states. However, wait-freedom ensures that termination happens after finitely many steps, so one of these states had to be univalent after all!

Now, let's think about what might happen from a bivalent critical state, S. WLOG, suppose that A's action leads to an outcome of 0 and B's to an outcome of 1. You should be able to convince yourself that A, B must be taking action on the same register - if A is acting on rA and B is acting on rB, then running A and then B or B and then from A from S leads to the same state, but one is 0 valent and the other is 1 valent! Similarly, both A, B can't be just doing reads. So this leads to two cases:

If A is writing X to r and B is reading r, then you can just run A in isolation from S and get 0. But if you let B just read r, and *then* run A in isolation from there, nothing about S that A can see has changed! So it must still give 0, but after letting B take a step from S you get 1! 

If A is writing X to r and B is writing Y to r, a similar problem happens. Just letting A run from S gives 0, but letting B take one step and then letting A run must give 1, and yet we know that A is just going to overwrite the value written to r by B and so must take the same steps as letting it run from S!

