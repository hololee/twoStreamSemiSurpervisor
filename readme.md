## Semi-supervised learning algorithm idea.

~~~
DA: labeled data , LA: label
DB:unlabeled data

modelA : initial stream A
modelB : initial stream B

streamA(n, m); n: number of updated by DA , m: number of updated by DB  = DA first updated
streamB(n, m); n: number of updated by DA , m: number of updated by DB  = DB first updated

========calculation=========
-----------------------------EPOCH----------------------------------
streamA(1, 0) = modelA.train(DA, LA)
streamB(0, 1) = modelB.train(DB, streamA(1, 0).predict(DB))
streamB(1, 1) = streamB(0, 1).train(DA, LA)
streamA(1, 1) = streamA(1, 0).train(DB, streamB(1, 1).predict(DB))
-----------------------------------------------------------------------
streamA(2, 1) = streamA(1, 1).train(DA, LA)
streamB(1, 2) = streamB(1, 1).train(DB, streamA(2, 1).predcit(DB))
streamB(2, 2) = streamB(1, 2).train(DA, LA)
streamA(2, 2) = streamA(2, 1).train(DB, streamB(2, 2).predict(DB))
-----------------------------------------------------------------------
streamA(3, 2) = streamA(2, 2).train(DA, LA)
streamB(2, 3) = streamB(2, 2).train(DB, streamA(3, 2).predcit(DB))
streamB(3, 3) = streamB(2, 3).train(DA, LA)
streamA(3, 3) = streamA(3, 2).train(DB, streamB(3, 3).predict(DB))
-----------------------------------------------------------------------
~~~

- key01 : 가짜 데이터에 의한 학습에는 가중치를 적게 두는게 어떤지?