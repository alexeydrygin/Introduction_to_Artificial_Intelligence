predicates
  nondeterm simpaty(symbol, symbol)
  nondeterm car(symbol)
  nondeterm color(symbol, symbol)
   
clauses
  % Что любит Сергей
  simpaty(sergey, bmw).
  simpaty(sergey, aeroplan).
  simpaty(sergey, toyota).

  % что любит Николай
  % Николай любит то, что любит Сергей, если это автомобиль и если он красный
  simpaty(nicolay, X):-simpaty(sergey, X), car(X), color(X, red).
  % Николай любит то, что любит Сергей, если это аэроплан
  simpaty(nicolay, X):-simpaty(sergey, X), X=aeroplan.

  % справочная информация
  car(bmw).  
  car(toyota).

  color(bmw, yellow).
  color(audi, orange).
  color(toyota, red).
  color(toyota, yellow).
   
goal
  simpaty(nicolay, X).