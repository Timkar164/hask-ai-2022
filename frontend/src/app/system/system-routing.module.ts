import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AuthComponent } from '../auth/auth.component';
import { SystemComponent } from './system.component';

const routes: Routes = [
  {
    path: '', component: SystemComponent, children: [
      { path: 'list', component: SystemComponent },
    ]
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class SystemRoutingModule { }
