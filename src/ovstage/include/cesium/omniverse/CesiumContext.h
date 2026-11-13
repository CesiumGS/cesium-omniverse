#pragma once

struct ovstage_instance_t;

class CesiumContext {
  public:
    CesiumContext(ovstage_instance_t* pStage);

  private:
    ovstage_instance_t* _pStage;
};
