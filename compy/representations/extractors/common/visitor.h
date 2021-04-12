#pragma once

namespace compy {

struct IVisitee;
struct IVisitor {
  virtual void visit(IVisitee* v) = 0;
};

struct IVisitee {
  virtual void accept(IVisitor* v) = 0;
};

}  // namespace compy